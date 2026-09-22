"""Whisper baseline on the same frozen test manifest, through the same harness as Dual-Fusion.

  torchrun --nproc_per_node=4 evaluations/inference.py \
      --manifest manifests/greek_fleurs_test.jsonl --out runs/greek_fleurs_test/whisper_plain --decoding plain

Whisper with its own recommended decoding (report as a separate "each system at its best" row):
      ... --decoding whisper_standard --out runs/greek_fleurs_test/whisper_standard

Plain `python evaluations/inference.py ...` also works (one GPU). Unlike the old script, this one
reads the rank from torchrun, so it no longer runs four racing copies of the evaluation.
"""
import argparse
import os
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from speechLM_utils.environment import set_environment

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
set_environment()

import torch
import transformers
from transformers import AutoProcessor, WhisperForConditionalGeneration

transformers.logging.set_verbosity_error()

from evaluations import fair_eval as fe

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class WhisperTranscriber:
    def __init__(self, model_name, language, device, dtype, decoding, max_new_tokens):
        self.device, self.language = device, language
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.special_ids = set(self.tokenizer.all_special_ids)  # <|endoftext|>, <|el|>, <|transcribe|>, ...
        self.generate_kwargs, self.make_processors, self.decoding_description = fe.resolve_decoding(
            decoding, "whisper", self.special_ids, max_new_tokens)
        self.max_new_tokens = self.generate_kwargs["max_new_tokens"]

    @torch.inference_mode()
    def transcribe(self, rows, arrays):
        features = fe.extract_features(self.processor.feature_extractor, arrays)
        output = self.model.generate(
            input_features=features.input_features.to(self.device, dtype=self.model.dtype),
            language=self.language, task="transcribe", return_timestamps=False,
            logits_processor=self.make_processors(), **self.generate_kwargs)
        sequences = output["sequences"] if isinstance(output, dict) else getattr(output, "sequences", output)
        results = []
        for ids in sequences.tolist():
            text_ids = [t for t in ids if t not in self.special_ids]
            results.append({"hypothesis_raw": self.tokenizer.decode(text_ids, skip_special_tokens=True).strip(),
                            "n_tokens": len(text_ids), "hit_cap": len(text_ids) >= self.max_new_tokens})
        return results


def resolve_language(requested, meta):
    manifest_language = meta.get("language_code")
    if requested and manifest_language and requested != manifest_language:
        raise ValueError(f"--language {requested} contradicts the manifest language ({manifest_language})")
    language = requested or manifest_language
    if language is None:
        raise ValueError("the manifest has no language code; pass --language el|hu")
    return language


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="built with fair_eval.py build-manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model_name", default="openai/whisper-large-v3")
    parser.add_argument("--language", default="el", choices=["el", "hu"],
                        help="forced decoding language; defaults to the manifest's language")
    parser.add_argument("--decoding", default="whisper_standard", choices=fe.presets_for("whisper"))
    parser.add_argument("--max_new_tokens", type=int, default=None, help="override the preset's cap")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPES),
                        help="bfloat16 matches the precision of the Dual-Fusion Whisper encoder")
    return parser.parse_args()  # strict: an old --gen_kwargs now fails instead of being ignored


def main():
    args = parse_args()
    _, _, local_rank = fe.dist_info()
    fe.prepare_out_dir(args.out, args.overwrite)
    rows, meta = fe.load_manifest(args.manifest)
    language = resolve_language(args.language, meta)

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    transcriber = WhisperTranscriber(args.model_name, language, device, DTYPES[args.dtype],
                                     args.decoding, args.max_new_tokens)
    run_config = fe.base_run_config(
        "whisper", args.manifest, meta, transcriber.decoding_description, args.batch_size,
        transcriber.processor.feature_extractor,
        model=args.model_name, dtype=args.dtype, forced_language=language)
    fe.run_sharded(transcriber, rows, meta, args.out, args.batch_size, run_config)


if __name__ == "__main__":
    main()
