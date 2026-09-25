"""Black-box Dual-Fusion ASR: load the trained model once, transcribe raw audio arrays.

The public object implements exactly the interface in sample_asr_input.py:

    asr = DualFusionASR()                       # the Greek model (MODEL_PRESETS in utils.py)
    asr = DualFusionASR(preset="hungarian")     # or the Hungarian one
    result = asr.transcribe(audio=[np_array, ...], batch_size=8, timestamps=True)
    # {"segments": [{"text", "score", "start", "end", "words": [...]}, ...], "language": "el"}

One segment is returned per input array, in the input order. Audio is resampled to 16 kHz if
needed and split into <=30 s chunks (the encoder window) at low-energy points; the chunk
transcripts of one input are joined into a single segment.

Model loading, decoding presets and the training-consistent position ids are imported from the
evaluation harness, so this service and the numbers in the paper come from the same code path.

CLI (mirrors transcribe_folder from sample_asr_input.py):

    python dual_fusion_asr.py /path/to/audio_folder --out transcripts.json
    python dual_fusion_asr.py /path/to/audio_folder --model hungarian --out transcripts.json

Any preset field can still be overridden (--exp, --machine, --datetime, --name,
--language_model_id, --train_datasets, --checkpoint).
"""
import argparse
import json
import os
import sys
import warnings
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from speechLM_utils.environment import set_environment

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
set_environment()

import numpy as np
import torch
import transformers
from transformers import GenerationConfig

transformers.logging.set_verbosity_error()

from utils import resolve_decoding, as_list, extract_features
from utils import DEFAULT_PRESET, MODEL_PRESETS, resolve_preset
from utils import (_unwrap, load_trained_weights, resolve_checkpoint, use_training_position_ids)
from utils import get_model
from utils import init

ENCODER_WINDOW_S = 30.0
TARGET_SR = 16_000


# ============================================================================ audio helpers
def to_mono_16k(audio, sample_rate):
    """Accept a 1-D or (channels, n) array at any rate; return mono float32 at 16 kHz."""
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=0)
    if sample_rate != TARGET_SR:
        import torchaudio

        tensor = torch.from_numpy(array).unsqueeze(0)
        array = torchaudio.functional.resample(tensor, sample_rate, TARGET_SR).squeeze(0).numpy()
    return array


def split_into_chunks(array, max_s=ENCODER_WINDOW_S, search_s=2.0, frame_s=0.02):
    """Split audio longer than the encoder window, cutting at the quietest frame near each
    boundary so that words are not cut in half. Returns [(chunk, offset_seconds), ...]."""
    limit = int(max_s * TARGET_SR)
    if len(array) <= limit:
        return [(array, 0.0)]

    chunks, start = [], 0
    while start < len(array):
        if len(array) - start <= limit:
            chunks.append((array[start:], start / TARGET_SR))
            break
        window_start = start + limit - int(search_s * TARGET_SR)
        frame = int(frame_s * TARGET_SR)
        window = array[window_start:start + limit]
        energies = [np.abs(window[i:i + frame]).mean() for i in range(0, max(len(window) - frame, 1), frame)]
        cut = window_start + (int(np.argmin(energies)) * frame if energies else len(window))
        cut = max(cut, start + frame)  # never produce an empty chunk
        chunks.append((array[start:cut], start / TARGET_SR))
        start = cut
    return chunks


# ============================================================================ word timings
def group_tokens_into_words(tokenizer, token_ids):
    """[(word, index of its first token), ...] using incremental decoding, so it works for both
    byte-level BPE (Llama/KriKri) and sentencepiece-style vocabularies (Qwen/Racka)."""
    words, text_so_far = [], ""
    for i in range(len(token_ids)):
        text = tokenizer.decode(token_ids[:i + 1], skip_special_tokens=True)
        piece = text[len(text_so_far):]
        text_so_far = text
        if not piece:
            continue
        if not words or piece[:1].isspace():
            words.append([piece.strip(), i])
        else:
            words[-1][0] += piece
    return [(w.strip(), i) for w, i in words if w.strip()]


def word_times_from_attention(words, frame_of_token, seconds_per_frame, offset, duration):
    """Turn per-token audio-frame attention peaks into monotone word start/end times."""
    timings, previous = [], offset
    for position, (word, token_index) in enumerate(words):
        if token_index < len(frame_of_token):
            start = offset + frame_of_token[token_index] * seconds_per_frame
        else:
            start = previous
        start = min(max(start, previous), offset + duration)
        next_index = words[position + 1][1] if position + 1 < len(words) else None
        if next_index is not None and next_index < len(frame_of_token):
            end = offset + frame_of_token[next_index] * seconds_per_frame
        else:
            end = offset + duration
        end = min(max(end, start), offset + duration)
        timings.append({"word": word, "start": round(float(start), 2), "end": round(float(end), 2)})
        previous = start
    return timings


def word_times_uniform(words, offset, duration):
    """Fallback when attention timings are unavailable: spread words over the chunk by length."""
    lengths = [max(len(w), 1) for w, _ in words]
    total = sum(lengths)
    timings, cursor = [], 0.0
    for (word, _), length in zip(words, lengths):
        start = offset + duration * cursor / total
        cursor += length
        end = offset + duration * cursor / total
        timings.append({"word": word, "start": round(float(start), 2), "end": round(float(end), 2)})
    return timings


# ============================================================================ the service
class DualFusionASR:
    """Loads the Dual-Fusion model once; `transcribe` is thread-unsafe but batch-friendly."""

    def __init__(self, preset=DEFAULT_PRESET, exp=None, machine=None, datetime=None, name=None,
                 turn=None, checkpoint=None, select="last", step=None, checkpoint_folder=None,
                 speech_encoder_id=None, language_model_id=None, train_datasets=None, language=None,
                 decoding="legacy", style_tag="off", position_ids="train",
                 max_new_tokens=None, sample_rate=TARGET_SR, device=None):
        # Everything about *which* model to load comes from the named preset; any argument that is
        # not None overrides it, so DualFusionASR() alone loads the default (Greek) model.
        cfg = resolve_preset(preset, exp=exp, machine=machine, datetime=datetime, name=name, turn=turn,
                             checkpoint=checkpoint, checkpoint_folder=checkpoint_folder,
                             speech_encoder_id=speech_encoder_id, language_model_id=language_model_id,
                             train_datasets=train_datasets, language=language)
        self.preset, self.config = preset, cfg
        self.language = cfg["language"]
        self.sample_rate = sample_rate
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        model_name = f"{cfg['speech_encoder_id'].split('/')[-1]}_{cfg['language_model_id'].split('/')[-1]}"
        info = {"res_folder": ".", "train_dataset": list(cfg["train_datasets"]),
                "checkpoint_folder": cfg["checkpoint_folder"], "model_name": model_name,
                "speech_encoder_id": cfg["speech_encoder_id"], "language_model_id": cfg["language_model_id"],
                "bit4": True, "machine": cfg["machine"], "datetime": cfg["datetime"], "turn": cfg["turn"],
                "exp": cfg["exp"], "name": cfg["name"],
                "gen_kwargs": None, "concat_test": False, "test_dataset": []}
        _, model_args, _, _, checkpoint_path, checkpoint_dir, _ = init(info, restart=False)

        explicit = cfg["checkpoint"] or checkpoint_dir
        try:  # the harness gained --step later; stay compatible with both versions
            resolved = resolve_checkpoint(checkpoint_path, explicit, select, step)
        except TypeError:
            resolved = resolve_checkpoint(checkpoint_path, explicit, select)
        self.checkpoint = resolved

        self.model, _ = get_model(model_args, info, self.device)
        self.tokenizer = self.model.language_tokenizer
        load_trained_weights(self.model, resolved)
        self.model.eval()
        if position_ids == "train":
            use_training_position_ids(self.model)

        base = _unwrap(self.model.language_model)
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_ids = sorted({self.tokenizer.eos_token_id, *as_list(base.generation_config.eos_token_id)})
        self.stop_ids = set(self.eos_ids) | {self.pad_id}
        special_ids = set(self.tokenizer.all_special_ids) | self.stop_ids
        clean = GenerationConfig(eos_token_id=self.eos_ids, pad_token_id=self.pad_id,
                                 bos_token_id=base.generation_config.bos_token_id)
        base.generation_config = clean
        self.model.language_model.generation_config = clean

        self.generate_kwargs, self.make_processors, self.decoding = resolve_decoding(
            decoding, "dual_fusion", special_ids, max_new_tokens)
        self.style_tag = None
        if style_tag not in (None, "off"):
            from speechLM_utils import data_collator as dc

            prompts = {"verbatim": getattr(dc, "PROMPT_VERBATIM", None), "formal": getattr(dc, "PROMPT_FORMAL", None),
                       "clean": getattr(dc, "PROMPT_CLEAN", None)}
            self.style_tag = prompts.get(style_tag)
            if self.style_tag is None:
                raise ValueError(f"style_tag '{style_tag}' is not defined for this language")
        self.alignment_module = self._find_alignment_module()

    # ------------------------------------------------------------------ internals
    def _find_alignment_module(self):
        layers = _unwrap(self.model.language_model).model.layers
        modules = [layer.cross_attention_layer for layer in layers if hasattr(layer, "cross_attention_layer")]
        return modules[len(modules) // 2] if modules else None  # middle depth aligns most sharply

    def _decode_ids(self, ids):
        start = 0  # generate() returns the pad-filled prompt followed by the new tokens
        while start < len(ids) and ids[start] == self.pad_id:
            start += 1
        generated = []
        for token in ids[start:]:
            if token in self.stop_ids:
                break
            generated.append(token)
        return generated

    @torch.inference_mode()
    def _run_batch(self, arrays, timestamps):
        features = extract_features(self.model.processor.feature_extractor, arrays)
        encoder_dtype = next(self.model.speech_encoder.parameters()).dtype
        inputs = {"audios": features.input_features.to(self.device, dtype=encoder_dtype),
                  "audio_masks": features.attention_mask.to(self.device)}
        if self.style_tag is not None:
            tags = self.tokenizer([self.style_tag] * len(arrays), return_tensors="pt",
                                  padding=True, add_special_tokens=False)
            inputs.update(tag_tokens=tags.input_ids, tag_masks=tags.attention_mask)

        kwargs = dict(self.generate_kwargs)
        capture = timestamps and self.alignment_module is not None
        if capture:  # attention rows must map 1:1 to batch items, so align with greedy decoding
            kwargs.update(num_beams=1, do_sample=False)
            self.alignment_module.attention_map = []
            self.alignment_module.store_attention = True

        self.model.logits_processor = self.make_processors()
        output = self.model.generate(**inputs, **kwargs)
        sequences = output["sequences"] if isinstance(output, dict) else getattr(output, "sequences", output)

        frames = self._peak_frames(len(arrays)) if capture else None
        if capture:
            self.alignment_module.store_attention = False
            self.alignment_module.attention_map = []
        return sequences.tolist(), frames

    def _peak_frames(self, batch_size):
        """Per batch item: the attended audio frame of each generated token, made monotone."""
        steps = [step for step in self.alignment_module.attention_map if step.shape[-2] == 1]
        if not steps:
            return None
        peaks, n_frames = [], steps[0].shape[-1]
        for item in range(batch_size):
            row, best = [], 0
            for step in steps:
                if item >= step.shape[0]:
                    break
                frame = int(step[item].float().mean(0)[0].argmax())
                best = max(best, frame)  # timings must not move backwards
                row.append(best)
            peaks.append(row)
        return peaks, n_frames

    # ------------------------------------------------------------------ public API
    def transcribe(self, audio, batch_size=32, timestamps=True, language=None, sample_rate=None):
        """audio: an iterable of mono numpy arrays (or one array). Returns the dict documented above."""
        if isinstance(audio, np.ndarray) and audio.ndim <= 1:
            audio = [audio]
        arrays = [to_mono_16k(a, sample_rate or self.sample_rate) for a in audio]
        if language and language != self.language:
            warnings.warn(f"this checkpoint transcribes {self.language}, not {language}")

        pieces = []  # (item index, chunk audio, offset in seconds)
        for index, array in enumerate(arrays):
            for chunk, offset in split_into_chunks(array):
                pieces.append((index, chunk, offset))

        outputs = [[] for _ in arrays]
        for start in range(0, len(pieces), batch_size):
            batch = pieces[start:start + batch_size]
            sequences, frames = self._run_batch([chunk for _, chunk, _ in batch], timestamps)
            for position, (index, chunk, offset) in enumerate(batch):
                token_ids = self._decode_ids(sequences[position])
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                duration = len(chunk) / TARGET_SR
                words = []
                if timestamps and text:
                    grouped = group_tokens_into_words(self.tokenizer, token_ids)
                    if frames and position < len(frames[0]) and frames[0][position]:
                        peaks, n_frames = frames[0][position], frames[1]
                        words = word_times_from_attention(grouped, peaks, ENCODER_WINDOW_S / n_frames,
                                                          offset, duration)
                    else:
                        words = word_times_uniform(grouped, offset, duration)
                outputs[index].append({"text": text, "words": words, "duration": duration, "offset": offset})

        segments = []
        for index, array in enumerate(arrays):
            parts = outputs[index]
            segments.append({
                "text": " ".join(part["text"] for part in parts if part["text"]).strip(),
                "score": 0.0,  # the model does not expose a calibrated confidence
                "start": 0.0,
                "end": round(len(array) / TARGET_SR, 2),
                "words": [word for part in parts for word in part["words"]],
            })
        return {"segments": segments, "language": language or self.language}


# ============================================================================ CLI
def transcribe_folder(audio_dir, asr, batch_size=8, timestamps=True):
    from pathlib import Path

    import torchaudio

    extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    paths = sorted(p for p in Path(audio_dir).iterdir() if p.is_file() and p.suffix.lower() in extensions)
    results = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        arrays = []
        for path in batch:
            waveform, rate = torchaudio.load(str(path))
            arrays.append(to_mono_16k(waveform.numpy(), rate))
        out = asr.transcribe(arrays, batch_size=batch_size, timestamps=timestamps)
        for path, segment in zip(batch, out["segments"]):
            results.append({"audio_path": str(path), **segment, "language": out["language"]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_dir")
    parser.add_argument("--out", default=None, help="write the results as JSON instead of printing")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_timestamps", action="store_true")
    parser.add_argument("--model", default=DEFAULT_PRESET, choices=list(MODEL_PRESETS),
                        help="which trained model to load (MODEL_PRESETS in utils.py)")
    parser.add_argument("--decoding", default="legacy")
    parser.add_argument("--style_tag", default="off")
    parser.add_argument("--position_ids", default="train", choices=["train", "hf"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step", type=int, default=None)
    # the rest default to the chosen --model preset
    parser.add_argument("--language", default=None, choices=["el", "hu"])
    parser.add_argument("--exp", type=int, default=None)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--datetime", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--language_model_id", default=None)
    parser.add_argument("--train_datasets", nargs="+", default=None)
    args = parser.parse_args()

    asr = DualFusionASR(preset=args.model, exp=args.exp, machine=args.machine, datetime=args.datetime,
                        name=args.name, checkpoint=args.checkpoint, step=args.step, language=args.language,
                        language_model_id=args.language_model_id, train_datasets=args.train_datasets,
                        decoding=args.decoding, style_tag=args.style_tag, position_ids=args.position_ids)
    results = transcribe_folder(args.audio_dir, asr, args.batch_size, not args.no_timestamps)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()