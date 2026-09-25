"""Black-box Greek ASR on top of a trained Dual-Fusion checkpoint.

Lives in product/ next to fair_speechLM.py, fair_eval.py, model.py, parser.py. The model is read from
product/checkpoints (next to this file, wherever you run it from; --checkpoint_dir overrides it):

    checkpoints/
        config.json          # from the training run folder
        checkpoint-XXXX/     # model.safetensors (or shards), trainer_state.json

Python, with the default model (legacy decoding, clean style):

    from speechLM_transcriber import DualFusionASR
    from sample_asr_input import transcribe_folder

    asr = DualFusionASR()
    results = transcribe_folder("/path/to/audio", asr, sample_rate=16000, batch_size=8)

Command line, with one audio file or a folder of audio files:

    python product/speechLM_transcriber.py /path/to/clip.wav
    python product/speechLM_transcriber.py /path/to/audio_dir --batch_size 8 --out results.jsonl

Add --manifest to score the transcripts against reference texts (WER / CER, printed and saved next to --out):

    python product/speechLM_transcriber.py /path/to/audio_dir --manifest refs.jsonl --out results.jsonl

Everything (weight loading, position-id fix, decoding presets, style tags) is the same code path as
fair_speechLM.py, so transcripts match the evaluation runs. Differences a user should know about:
  * audio longer than 30 s is cut into <=30 s chunks at quiet points and the texts are joined;
  * the model has no word timestamps, so "words" is always [] and "score" is always 0.0;
  * the model is Greek-only; `language` is only echoed back.
"""
import json
import os
import sys
import warnings
from os.path import abspath, dirname
import logging
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)
from environment import set_environment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
set_environment()

import numpy as np
import torch
import torchaudio.functional as AF

import fair_eval as fe
from fair_speechLM import (DualFusionTranscriber, load_trained_weights, resolve_checkpoint, use_training_position_ids)
from model import get_model
from utils import init_gpu
from parser import Parser

# ------------------------------------------------------------------ default model
DEFAULT_CHECKPOINT_DIR = os.path.join('.', "checkpoints")
DEFAULT_DECODING = "legacy"
DEFAULT_STYLE = "clean"
STYLE_ALIASES = {"clean": "formal"}  # clean = normalized transcription prompt (PROMPT_FORMAL)
SPEECH_ENCODER_ID = "openai/whisper-large-v3"
LANGUAGE_MODEL_ID = "ilsp/Llama-Krikri-8B-Instruct"

MODEL_SR = fe.SAMPLING_RATE  # 16 kHz, what Whisper's feature extractor expects
CHUNK_S = fe.MAX_AUDIO_S     # the encoder sees at most 30 s at once
SEARCH_S = 5.0               # look for a quiet cut point in the last 5 s of each chunk


def split_long_audio(wave, sr=MODEL_SR):
    """Split into <=30 s chunks, cutting at the quietest 20 ms frame of the last 5 s of each chunk."""
    max_len, search, frame = int(CHUNK_S * sr), int(SEARCH_S * sr), int(0.02 * sr)
    chunks, start = [], 0
    while len(wave) - start > max_len:
        window = wave[start + max_len - search:start + max_len]
        n = len(window) // frame
        energy = (window[:n * frame].reshape(n, frame) ** 2).mean(axis=1)
        cut = start + max_len - search + int(energy.argmin()) * frame
        chunks.append(wave[start:cut])
        start = cut
    chunks.append(wave[start:])
    return chunks


def load_model_args(checkpoint_dir):
    """YAML defaults of `exp`, overridden by the run's config.json (same rule as utils.init)."""
    config_file = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found; copy config.json from the training run folder")
    conf = Parser()
    conf.get_args()
    args = conf.dual_fuse_args
    with open(config_file, encoding="utf-8") as f:
        args.__dict__.update(json.load(f))
    return args


class DualFusionASR:
    def __init__(self,
                 checkpoint_dir=DEFAULT_CHECKPOINT_DIR,  # folder with config.json + checkpoint-N/
                 select="last",               # "last" or "best" (from trainer_state.json)
                 decoding=DEFAULT_DECODING,   # "legacy", "plain" or "penalized"
                 style=DEFAULT_STYLE,         # "clean"/"formal", "verbatim", "auto" or "off"
                 sample_rate=16000,           # rate of the arrays you will pass (16000 or 24000)
                 ):
        if sample_rate not in (16000, 24000):
            raise ValueError("sample_rate must be 16000 or 24000")
        self.sample_rate = sample_rate

        _, device = init_gpu()
        model_args = load_model_args(checkpoint_dir)
        info = {"speech_encoder_id": SPEECH_ENCODER_ID, "language_model_id": LANGUAGE_MODEL_ID}
        model, _ = get_model(model_args, info, device)

        self.checkpoint = abspath(resolve_checkpoint(checkpoint_dir, select))
        load_trained_weights(model, self.checkpoint)
        model.eval()
        use_training_position_ids(model)

        style = STYLE_ALIASES.get(style, style)
        trained_with_tags = getattr(model, "prompt_verbatim", False)
        if style == "auto":
            style = "verbatim" if trained_with_tags else "off"
        if style not in ("verbatim", "formal", "off"):
            raise ValueError("style must be 'clean', 'formal', 'verbatim', 'auto' or 'off'")
        if style != "off" and not trained_with_tags:
            raise ValueError("this checkpoint was trained without style tags; use style='off'")

        self._engine = DualFusionTranscriber(model, model.language_tokenizer, device, decoding, None, style)

        print(f"DualFusionASR ready: {self.checkpoint} | decoding={decoding} | style={style}")

    def _to_model_rate(self, wave, sr):
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1:
            raise ValueError("expected mono 1-D arrays")
        if sr != MODEL_SR:
            wave = AF.resample(torch.from_numpy(wave), sr, MODEL_SR).numpy()
        return wave

    def transcribe(self, audio, batch_size=4, timestamps=True, language="el", sample_rate=None):
        if language != "el":
            warnings.warn(f"the model only transcribes Greek; language='{language}' is ignored")
        if timestamps:
            warnings.warn("the model does not produce word timestamps; 'words' will be empty")

        waves = [self._to_model_rate(w, sample_rate or self.sample_rate) for w in audio]

        # Flatten every file into <=30 s chunks and remember which file each chunk belongs to.
        chunks, owner = [], []
        for i, wave in enumerate(waves):
            for chunk in split_long_audio(wave):
                if len(chunk):
                    chunks.append(chunk)
                    owner.append(i)

        texts = [[] for _ in waves]
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            outputs = self._engine.transcribe([{"dataset_name": None}] * len(batch), batch)
            for i, out in zip(owner[start:start + batch_size], outputs):
                if out["hit_cap"]:
                    warnings.warn(f"input {i}: output hit the token cap and may be truncated")
                if out["hypothesis_raw"]:
                    texts[i].append(out["hypothesis_raw"])

        segments = [{"text": " ".join(t), "score": 0.0, "start": 0.0,
                     "end": len(w) / MODEL_SR, "words": []}
                    for t, w in zip(texts, waves)]
        return {"segments": segments, "language": language}


# ------------------------------------------------------------------ scoring against a manifest
def load_references(manifest_path):
    """Read a JSON-lines (or JSON list) manifest and return a lookup: audio path -> reference.

    Each entry needs an audio path ("audio_filepath" or "audio_path") and a reference text
    ("reference_raw" as in fair_eval manifests, "text" as in NeMo manifests, or "reference").
    Relative audio paths are taken relative to the manifest's folder. If a full path does not
    match, a file is matched by its name alone, as long as that name is unique in the manifest."""
    base = dirname(abspath(manifest_path))
    with open(manifest_path, encoding="utf-8") as f:
        content = f.read().strip()
    entries = json.loads(content) if content.startswith("[") else \
        [json.loads(line) for line in content.splitlines() if line.strip()]

    by_path, by_name = {}, {}
    for entry in entries:
        audio = entry.get("audio_filepath") or entry.get("audio_path")
        reference = next((entry[k] for k in ("reference_raw", "text", "reference") if entry.get(k) is not None), None)
        if audio is None or reference is None:
            continue
        item = {"reference": reference, "duration": entry.get("duration")}
        by_path[os.path.normpath(os.path.join(base, os.path.expanduser(audio)))] = item
        by_name.setdefault(os.path.basename(audio), []).append(item)

    def lookup(audio_path):
        hit = by_path.get(os.path.normpath(abspath(os.path.expanduser(audio_path))))
        if hit is not None:
            return hit
        same_name = by_name.get(os.path.basename(audio_path), [])
        return same_name[0] if len(same_name) == 1 else None

    return lookup


def score_against_manifest(results, manifest_path):
    """Add "reference" and per-file "wer" to every result found in the manifest and return the
    corpus-level metrics (same normalization and numbers as fair_eval), or None if nothing matched."""
    lookup = load_references(manifest_path)
    matched, records = [], []
    for row in results:
        ref = lookup(row["audio_path"])
        if ref is None:
            continue
        row["reference"] = ref["reference"]
        matched.append(row)
        records.append({"reference_raw": ref["reference"], "hypothesis_raw": row["text"],
                        "duration": row.get("duration") or ref["duration"] or 0.0,
                        "hit_cap": row.get("hit_cap", False)})

    missing = len(results) - len(matched)
    if missing:
        print(f"WARNING: {missing} of {len(results)} files have no entry in {manifest_path} and are not scored")
    if not matched:
        return None

    for row, (sub, dele, ins, n_ref) in zip(matched, fe.utterance_table(records)["n_wer"]):
        row["wer"] = round(float(sub + dele + ins) / n_ref, 4) if n_ref else None
    return fe.score_records(records)


if __name__ == "__main__":
    import argparse

    # the coworker's file; must be importable (same folder, repo root, or PYTHONPATH)
    from sample_asr_input import iter_audio_batches, load_audio_as_numpy

    p = argparse.ArgumentParser()
    p.add_argument("path", help="one audio file or a folder of audio files")
    p.add_argument("--out", default=None, help="optional .jsonl file for the results")
    p.add_argument("--manifest", default=None, help="optional .jsonl with reference texts; enables scoring")
    p.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--decoding", default=DEFAULT_DECODING, choices=fe.presets_for("dual_fusion"))
    p.add_argument("--style", default=DEFAULT_STYLE, choices=["clean", "formal", "verbatim", "auto", "off"])
    p.add_argument("--sample_rate", type=int, default=16000, choices=[16000, 24000])
    p.add_argument("--batch_size", type=int, default=4)
    a = p.parse_args()

    asr = DualFusionASR(a.checkpoint_dir, decoding=a.decoding, style=a.style, sample_rate=a.sample_rate)

    if os.path.isdir(a.path):  # batch: every audio file in the folder
        batches = iter_audio_batches(a.path, sample_rate=a.sample_rate, batch_size=a.batch_size)
    else:                      # single file
        batches = [([a.path], [load_audio_as_numpy(a.path, sample_rate=a.sample_rate)])]

    out_file = open(a.out, "w", encoding="utf-8") if a.out else None
    results = []
    for paths, waves in batches:
        out = asr.transcribe(waves, batch_size=a.batch_size, timestamps=False)
        for path, seg in zip(paths, out["segments"]):
            results.append({"audio_path": str(path), "text": seg["text"], "score": seg["score"],
                            "words": seg["words"], "language": out["language"],
                            "duration": round(seg["end"], 3), "hit_cap": seg["hit_cap"]})
            if out_file:
                out_file.write(json.dumps(results[-1], ensure_ascii=False) + "\n")
                out_file.flush()
    if out_file:
        out_file.close()

    metrics = score_against_manifest(results, a.manifest) if a.manifest else None

    lines = [json.dumps(r, ensure_ascii=False) for r in results]
    if a.out:
        if metrics is not None:  # rewrite with the added "reference" and "wer" fields
            with open(a.out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} results to {a.out}")
    else:
        print("\n".join(lines))

    if metrics is not None:
        fe.print_metrics(f"scored against {a.manifest}", metrics)
        if a.out:
            metrics_file = os.path.splitext(a.out)[0] + ".metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"wrote metrics to {metrics_file}")