"""
Minimal ASR input snippet.

Given a folder of audio files, this loads each file as mono numpy audio at
16000 Hz or 24000 Hz, batches the arrays, and feeds them to a dummy transcriber.

Expected transcriber return shape:

    {
        "segments": [
            {
                "text": "recognized text",
                "score": 0.0,
                "words": [
                    {"word": "recognized", "start": 0.00, "end": 0.35},
                    {"word": "text", "start": 0.36, "end": 0.70},
                ],
            },
            ...
        ],
        "language": "el",
    }

The order of `segments` must match the order of input audio arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torchaudio


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_audio_as_numpy(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    if sample_rate not in (16000, 24000):
        raise ValueError("sample_rate must be 16000 or 24000")

    audio, original_sr = torchaudio.load(str(path))

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if original_sr != sample_rate:
        audio = torchaudio.transforms.Resample(
            orig_freq=original_sr,
            new_freq=sample_rate,
        )(audio)

    return audio.squeeze(0).numpy()


def iter_audio_batches(
    audio_dir: str | Path,
    sample_rate: int = 16000,
    batch_size: int = 32,
):
    audio_dir = Path(audio_dir)
    audio_paths = sorted(
        path
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    batch_paths = []
    batch_audio = []

    for path in audio_paths:
        batch_paths.append(path)
        batch_audio.append(load_audio_as_numpy(path, sample_rate=sample_rate))

        if len(batch_audio) == batch_size:
            yield batch_paths, batch_audio
            batch_paths = []
            batch_audio = []

    if batch_audio:
        yield batch_paths, batch_audio


class DummyTranscriber:
    def transcribe(
        self,
        audio: Iterable[np.ndarray],
        batch_size: int = 32,
        timestamps: bool = True,
        language: str = "el",
    ) -> dict:
        segments = []
        for idx, waveform in enumerate(audio):
            duration = len(waveform) / 16000.0
            segments.append(
                {
                    "text": f"dummy transcript {idx}",
                    "score": 0.0,
                    "start": 0.0,
                    "end": duration,
                    "words": [
                        {"word": "dummy", "start": 0.0, "end": min(0.3, duration)},
                        {"word": "transcript", "start": min(0.3, duration), "end": duration},
                    ] if timestamps else [],
                }
            )

        return {
            "segments": segments,
            "language": language,
        }


def transcribe_folder(
    audio_dir: str | Path,
    transcriber,
    sample_rate: int = 16000,
    batch_size: int = 32,
    language: str = "el",
) -> list[dict]:
    all_results = []

    for batch_paths, batch_audio in iter_audio_batches(
        audio_dir,
        sample_rate=sample_rate,
        batch_size=batch_size,
    ):
        result = transcriber.transcribe(
            audio=batch_audio,
            batch_size=batch_size,
            timestamps=True,
            language=language,
        )

        for path, segment in zip(batch_paths, result["segments"]):
            all_results.append(
                {
                    "audio_path": str(path),
                    "text": segment.get("text", ""),
                    "score": segment.get("score", 0.0),
                    "words": segment.get("words", []),
                    "language": result.get("language", language),
                }
            )

    return all_results


if __name__ == "__main__":
    folder = "/path/to/audio_folder"
    transcriber = DummyTranscriber()
    results = transcribe_folder(
        folder,
        transcriber,
        sample_rate=16000,  # or 24000
        batch_size=32,
        language="el",
    )
    for item in results:
        print(item)
