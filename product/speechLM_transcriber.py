"""Black-box Greek ASR on top of a trained Dual-Fusion checkpoint.

Put this file in evaluations/ (next to fair_speechLM.py). The model is read from ./checkpoints
(in the repo root), which must look like this:

    checkpoints/
        config.json          # from the training run folder
        checkpoint-XXXX/     # model.safetensors (or shards), trainer_state.json

Python, with the default model (exp 1, legacy decoding, clean style):

    from evaluations.speechLM_transcriber import DualFusionASR
    from sample_asr_input import transcribe_folder

    asr = DualFusionASR()
    results = transcribe_folder("/path/to/audio", asr, sample_rate=16000, batch_size=8)

Command line, with one audio file or a folder of audio files:

    python evaluations/speechLM_transcriber.py /path/to/clip.wav
    python evaluations/speechLM_transcriber.py /path/to/audio_dir --batch_size 8 --out results.jsonl

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
from utils import init, init_gpu
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

    def transcribe(self, audio, batch_size=32, timestamps=True, language="el", sample_rate=None):
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


if __name__ == "__main__":
    import argparse

    # the coworker's file; must be importable (same folder, repo root, or PYTHONPATH)
    from sample_asr_input import load_audio_as_numpy, transcribe_folder

    p = argparse.ArgumentParser()
    p.add_argument("path", help="one audio file or a folder of audio files")
    p.add_argument("--out", default=None, help="optional .jsonl file for the results")
    p.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--decoding", default=DEFAULT_DECODING, choices=fe.presets_for("dual_fusion"))
    p.add_argument("--style", default=DEFAULT_STYLE, choices=["clean", "formal", "verbatim", "auto", "off"])
    p.add_argument("--sample_rate", type=int, default=16000, choices=[16000, 24000])
    p.add_argument("--batch_size", type=int, default=4)
    a = p.parse_args()

    asr = DualFusionASR(a.checkpoint_dir, decoding=a.decoding, style=a.style, sample_rate=a.sample_rate)

    if os.path.isdir(a.path):  # batch: every audio file in the folder
        results = transcribe_folder(a.path, asr, sample_rate=a.sample_rate, batch_size=a.batch_size)
    else:                      # single file, same output format
        wave = load_audio_as_numpy(a.path, sample_rate=a.sample_rate)
        out = asr.transcribe([wave], batch_size=1)
        seg = out["segments"][0]
        results = [{"audio_path": a.path, "text": seg["text"], "score": seg["score"],
                    "words": seg["words"], "language": out["language"]}]

    lines = [json.dumps(r, ensure_ascii=False) for r in results]
    print("\n".join(lines))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} results to {a.out}")