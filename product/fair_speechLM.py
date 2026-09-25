"""Dual-Fusion evaluation on a frozen test manifest, through the shared fair-comparison harness.

  torchrun --nproc_per_node=4 evaluations/speechLM.py \
      --manifest manifests/greek_fleurs_test.jsonl --out runs/greek_fleurs_test/dual_fusion_plain \
      --exp 1 --machine greekasr-0 --datetime Sep10_13-36 --name full --decoding plain

Reproduce the settings behind the current paper numbers (old decoding, no style tag at test
time, Hugging Face's attention-mask-derived position ids):
      ... --decoding legacy --style_tags off --position_ids hf --out runs/greek_fleurs_test/dual_fusion_legacy
"""
import argparse
import json
import os
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from environment import set_environment

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
set_environment()

import safetensors.torch
import torch
import transformers
from transformers import GenerationConfig

transformers.logging.set_verbosity_error()

import fair_eval as fe
from model import get_max_step

PROMPT_VERBATIM = "Κατά λέξη μεταγραφή: "
PROMPT_FORMAL = "Κανονικοποιημένη μεταγραφή: "

DATASET_PROMPT_MAP = {
    'common_voice': PROMPT_VERBATIM,
    'fleurs': PROMPT_VERBATIM,
    'logotypographia': PROMPT_VERBATIM,
    'stoma': PROMPT_VERBATIM
}

# Parameters that were trained (bridge + LoRA) and therefore must come from the checkpoint.
TRAINED_PARAM_MARKERS = ("lora_", "adapter.", "input_downsampler", "injection_downsampler",
                         "cross_attention_layer", "layer_dynamic", "layer_static")
# bitsandbytes quantization state of the frozen base LLM; reloaded from the Hub, never needed.
QUANT_STATE_MARKERS = ("absmax", "quant_map", "quant_state", "bitsandbytes", "nested_")


# ============================================================================ checkpoint
def resolve_checkpoint(checkpoint_path, select):
    last = os.path.join(checkpoint_path, f"checkpoint-{get_max_step(checkpoint_path)}")
    if select == "last":
        return last
    with open(os.path.join(last, "trainer_state.json"), encoding="utf-8") as f:
        best = json.load(f).get("best_model_checkpoint")
    if not best:
        raise RuntimeError(f"no best_model_checkpoint recorded in {last}/trainer_state.json")
    return os.path.join(checkpoint_path, os.path.basename(best.rstrip("/")))


def read_state_dict(checkpoint):
    index_file = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, encoding="utf-8") as f:
            shards = set(json.load(f)["weight_map"].values())
        state = {}
        for shard in shards:
            state.update(safetensors.torch.load_file(os.path.join(checkpoint, shard)))
        return state
    if os.path.exists(os.path.join(checkpoint, "model.safetensors")):
        return safetensors.torch.load_file(os.path.join(checkpoint, "model.safetensors"))
    if os.path.exists(os.path.join(checkpoint, "pytorch_model.bin")):
        return torch.load(os.path.join(checkpoint, "pytorch_model.bin"), map_location="cpu")
    raise FileNotFoundError(f"no weights found in {checkpoint}")


def load_trained_weights(model, checkpoint):
    """Fail loudly instead of silently evaluating a randomly initialized bridge."""
    state = {k: v for k, v in read_state_dict(checkpoint).items()
             if not any(m in k for m in QUANT_STATE_MARKERS)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    trained = {n for n, _ in model.named_parameters() if any(m in n for m in TRAINED_PARAM_MARKERS)}
    if not trained:
        raise RuntimeError("no bridge/LoRA parameters found; update TRAINED_PARAM_MARKERS")
    not_loaded = sorted(trained & set(missing))
    if not_loaded or unexpected:
        raise RuntimeError(f"checkpoint {checkpoint} does not match the model: "
                           f"{len(not_loaded)} trained params missing (e.g. {not_loaded[:3]}), "
                           f"{len(unexpected)} unexpected keys (e.g. {list(unexpected)[:3]})")
    print(f"[rank {fe.dist_info()[0]}] loaded {len(trained)} trained tensors from {checkpoint}")


# ============================================================================ model patches
def _unwrap(language_model):
    return language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model


def use_training_position_ids(model):
    """Training called the LLM without position_ids, so RoPE positions were 0..L-1 *including* the
    masked (padded) prompt-audio slots. generate() instead derives position ids from
    attention_mask.cumsum(), which skips those slots and shifts every text-to-audio distance.
    Dropping position_ids makes the decoder fall back to cache_position, i.e. exactly the
    training positions, at every generation step."""
    inner = _unwrap(model.language_model).model
    original_forward = inner.forward

    def forward(*args, **kwargs):
        kwargs["position_ids"] = None
        return original_forward(*args, **kwargs)

    inner.forward = forward


# ============================================================================ transcriber
class DualFusionTranscriber:
    def __init__(self, model, tokenizer, device, decoding, max_new_tokens, style_tags):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.style_tags = style_tags
        self.pad_id = tokenizer.pad_token_id

        base = _unwrap(model.language_model)
        hub = base.generation_config
        self.eos_ids = sorted({tokenizer.eos_token_id, *fe.as_list(hub.eos_token_id)})
        self.stop_ids = set(self.eos_ids) | {self.pad_id}
        special_ids = set(tokenizer.all_special_ids) | self.stop_ids

        # Replace the Hub generation config (instruct models ship sampling / top_p / penalties)
        # with an empty one; every decoding parameter is then passed explicitly below.
        clean = GenerationConfig(eos_token_id=self.eos_ids, pad_token_id=self.pad_id,
                                 bos_token_id=hub.bos_token_id)
        base.generation_config = clean
        model.language_model.generation_config = clean

        self.generate_kwargs, self.make_processors, self.decoding_description = fe.resolve_decoding(
            decoding, "dual_fusion", special_ids, max_new_tokens)
        self.max_new_tokens = self.generate_kwargs["max_new_tokens"]

    def _tag(self, row):
        if self.style_tags == "dataset":  # the same corpus -> tag rule as the training collator
            return DATASET_PROMPT_MAP.get(row["dataset_name"], PROMPT_FORMAL)
        return PROMPT_VERBATIM if self.style_tags == "verbatim" else PROMPT_FORMAL

    def _decode(self, ids):
        start = 0  # generate() returns the pad-filled prompt ids followed by the new tokens
        while start < len(ids) and ids[start] == self.pad_id:
            start += 1
        generated = []
        for token in ids[start:]:
            if token in self.stop_ids:
                break
            generated.append(token)
        return {"hypothesis_raw": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                "n_tokens": len(generated), "hit_cap": len(generated) >= self.max_new_tokens}

    @torch.inference_mode()
    def transcribe(self, rows, arrays):
        features = fe.extract_features(self.model.processor.feature_extractor, arrays)
        encoder_dtype = next(self.model.speech_encoder.parameters()).dtype
        inputs = {"audios": features.input_features.to(self.device, dtype=encoder_dtype),
                  "audio_masks": features.attention_mask.to(self.device)}
        if self.style_tags != "off":
            tags = self.tokenizer([self._tag(r) for r in rows], return_tensors="pt",
                                  padding=True, add_special_tokens=False)
            inputs.update(tag_tokens=tags.input_ids, tag_masks=tags.attention_mask)
        # DualFusionModel.generate() hands self.logits_processor to the LLM's generate().
        self.model.logits_processor = self.make_processors()
        sequences = self.model.generate(**inputs, **self.generate_kwargs)
        return [self._decode(ids) for ids in sequences.tolist()]


# ============================================================================ main
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="built with fair_eval.py build-manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--decoding", default="legacy", choices=fe.presets_for("dual_fusion"))
    parser.add_argument("--max_new_tokens", type=int, default=None, help="override the preset's cap")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--style_tags", default="auto", choices=["auto", "dataset", "verbatim", "formal", "off"],
                        help="auto = per-corpus tags if the model was trained with prompt_verbatim, else off")
    parser.add_argument("--position_ids", default="train", choices=["train", "hf"],
                        help="train = RoPE positions as in training; hf = the old generate() behaviour")
    parser.add_argument("--checkpoint", default=None, help="explicit checkpoint directory")
    parser.add_argument("--select", default="last", choices=["last", "best"])
    # identify the trained run (same meaning as before)
    parser.add_argument("--exp", type=int, default=0)
    parser.add_argument("--machine", default="kronos")
    parser.add_argument("--datetime", default=None)
    parser.add_argument("--turn", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--checkpoint_folder", default="dual_fusion_checkpoints")
    parser.add_argument("--speech_encoder_id", default="openai/whisper-large-v3")
    parser.add_argument("--language_model_id", default="ilsp/Llama-Krikri-8B-Instruct")
    parser.add_argument("--train_datasets", nargs="+",
                        default=["common_voice", "fleurs", "hparl", "tedx", "logotypographia"])
    return parser.parse_args()  # strict: an old --gen_kwargs now fails instead of being ignored