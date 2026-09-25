# -------------------------------------------------------------------------------------------------------------------
# EVALUATION UTILS
# -------------------------------------------------------------------------------------------------------------------


import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from importlib.metadata import version as package_version
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))
try:  # keep the project's cache/env conventions; must run before any Hugging Face import
    from speechLM_utils.environment import set_environment

    set_environment()
except ImportError:
    pass

import jiwer
import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

SAMPLING_RATE = 16_000
MAX_AUDIO_S = 30.0
LANG_CODES = {"greek": "el", "hungarian": "hu"}

# ============================================================================ model presets
# Ready-made model configurations, so callers only choose a language. `train_datasets` must list
# the training corpora exactly as the checkpoint folder names them (same names, same order);
# every field can be overridden per call.
MODEL_PRESETS = {
    "greek": {
        "language": "el",
        "exp": 1,
        "machine": "greekasr-0",
        "datetime": "Sep10_13-36",
        "name": "full",
        "language_model_id": "ilsp/Llama-Krikri-8B-Instruct",
        "train_datasets": ["common_voice", "fleurs", "hparl", "tedx", "logotypographia", "stoma"],
    },
    "hungarian": {
        "language": "hu",
        "exp": 6,
        "machine": "hu-asr-medium-3-0",
        "datetime": "Sep10_10-23",
        "name": "full",
        "language_model_id": "elte-nlp/Racka-4B",
        "train_datasets": ["common_voice", "fleurs", "speech_massive", "voxpopuli", "yodas"],
    },
}
DEFAULT_PRESET = "greek"
COMMON_DEFAULTS = {"turn": None, "checkpoint": None, "checkpoint_folder": "dual_fusion_checkpoints",
                   "speech_encoder_id": "openai/whisper-large-v3"}


def resolve_preset(preset=DEFAULT_PRESET, **overrides):
    """Merge a named model preset with explicitly given values; None means 'use the preset'."""
    if preset is not None and preset not in MODEL_PRESETS:
        raise ValueError(f"unknown preset '{preset}'; available: {', '.join(MODEL_PRESETS)}")
    settings = dict(COMMON_DEFAULTS)
    settings.update(MODEL_PRESETS[preset] if preset else {})
    settings.update({key: value for key, value in overrides.items() if value is not None})
    required = ("exp", "machine", "datetime", "language", "language_model_id", "train_datasets")
    missing = [key for key in required if settings.get(key) is None]
    if missing and not settings.get("checkpoint"):
        raise ValueError(f"no preset selected and nothing given for: {', '.join(missing)}")
    return settings

SPEAKER_TAGS = ("ομιλητής", "ομιλήτρια", "speaker", "beszélő")

def _fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn").casefold()

_SPEAKER_TAG_RE = re.compile(r"^\s*(?:%s)\s*\d*\s*:\s*" % "|".join(re.escape(_fold(t)) for t in SPEAKER_TAGS))
_BRACKETS_RE = re.compile(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)")
_QUOTES = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})
_BASIC = BasicTextNormalizer()


def strip_speaker_tag(text):
    folded = _fold(text)
    if len(folded) != len(text):  # folding changed the length: fall back to a plain casefold
        folded = text.casefold()
        if len(folded) != len(text):
            return text
    match = _SPEAKER_TAG_RE.match(folded)
    return text[match.end():] if match else text


def _cleanup(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = strip_speaker_tag(text)
    text = _BRACKETS_RE.sub(" ", text)
    return text.lower().translate(_QUOTES)

import unicodedata
strip_accents = lambda s: unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"))
STRIP_ACCENTS = False

def normalize_nwer(text):
    text = _BASIC(_cleanup(text)).strip()
    if STRIP_ACCENTS:
        text = strip_accents(text)
    return text

def normalize_wer_punct(text):
    text = re.sub(r"([^\w\s])\1*", r" \g<0> ", _cleanup(text))
    return re.sub(r"\s+", " ", text).strip()

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _write_jsonl(path, records):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=list)
    os.replace(tmp, path)

def as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def build_manifest(exp, datasets, split, out_path, check_train_overlap=True):
    from config.parser import Parser
    from preprocessing.prepare import get_data
    from preprocessing.split import splitter

    conf = Parser()
    conf.get_args(exp)
    splits = splitter(conf=conf).split(datasets=list(datasets))
    raw = get_data(splits[split], "", process=False, filters=None, split=split)

    rows, dropped = [], []
    for name, items in raw.items():
        for item in items:
            reference = item.get("text") or ""
            row = {"id": f"{name}:{item['index']}", "dataset_name": name, "index": int(item["index"]),
                   "audio_filepath": item["audio_filepath"], "reference_raw": reference,
                   "duration": float(item.get("duration") or 0.0)}
            keep = normalize_nwer(reference) and normalize_wer_punct(reference)
            (rows if keep else dropped).append(row)
    rows.sort(key=lambda r: (r["dataset_name"], r["index"]))
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate utterance ids in the requested split")

    os.makedirs(dirname(abspath(out_path)), exist_ok=True)
    _write_jsonl(out_path, rows)
    language = str(getattr(conf, "language", "")).lower()
    meta = {"exp": exp, "datasets": list(datasets), "split": split,
            "language": language, "language_code": LANG_CODES.get(language),
            "n_rows": len(rows), "n_over_30s": sum(r["duration"] > MAX_AUDIO_S for r in rows),
            "n_dropped_empty_reference": len(dropped), "dropped_ids": [r["id"] for r in dropped],
            "sha256": _sha256(out_path)}

    if check_train_overlap and split != "train":
        # Speaker-disjoint splits are not text-disjoint; an LLM decoder can exploit repeated sentences.
        train = get_data(splits["train"], "", process=False, filters=None, split="train")
        train_texts = {normalize_nwer(it.get("text") or "") for items in train.values() for it in items}
        overlap = [r["id"] for r in rows if normalize_nwer(r["reference_raw"]) in train_texts]
        meta["n_reference_text_also_in_train"] = len(overlap)
        meta["reference_text_also_in_train_ids"] = overlap

    _write_json(f"{out_path}.meta.json", meta)
    return meta


def load_manifest(path):
    with open(f"{path}.meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    if _sha256(path) != meta["sha256"]:
        raise RuntimeError(f"{path} changed after it was built; rebuild it and rerun every system")
    return _read_jsonl(path), meta


# ============================================================================ audio
def _to_float_array(audio):
    if isinstance(audio, dict):  # datasets < 4
        return np.asarray(audio["array"], dtype=np.float32)
    samples = audio.get_all_samples()  # datasets >= 4 returns a torchcodec AudioDecoder
    return samples.data.mean(dim=0).numpy().astype(np.float32)


def iter_audio_batches(rows, batch_size):
    """Decode and resample through the same datasets.Audio path that training used."""
    from datasets import Audio, Dataset

    dataset = Dataset.from_dict({"audio": [r["audio_filepath"] for r in rows]})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    for start in range(0, len(rows), batch_size):
        audio = dataset[start:start + batch_size]["audio"]
        yield rows[start:start + batch_size], [_to_float_array(a) for a in audio]


def extract_features(feature_extractor, arrays):
    """Identical input for both systems: log-mel of the first 30 s, padded to 30 s."""
    return feature_extractor(arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt",
                             padding="max_length", truncation=True, return_attention_mask=True)


# ============================================================================ decoding
# "penalties":
#   none           -> no repetition penalty / n-gram blocking at all
#   generated_only -> our processors below: only generated text tokens, never prompt or EOS
#   builtin        -> Hugging Face's own processors, exactly as in the runs behind the paper
DECODING_PRESETS = {
    # Primary, controlled comparison. Token-level penalties are left out on purpose: their effect
    # depends on how finely each tokenizer splits the language, so identical values are not
    # identical treatment. The token cap is one that no reference should reach.
    "plain": {"num_beams": 5, "do_sample": False, "length_penalty": 1.0, "max_new_tokens": 440,
              "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "penalties": "none"},
    # Paper-style penalties restricted to generated text tokens. Still tokenizer-dependent:
    # report only as an "each system with penalties" row, never as the controlled comparison.
    "penalized": {"num_beams": 5, "do_sample": False, "length_penalty": 1.0, "max_new_tokens": 440,
                  "repetition_penalty": 1.15, "no_repeat_ngram_size": 4, "penalties": "generated_only"},
    # Exactly the settings behind the current paper numbers (HF built-ins, 150-token cap).
    "legacy": {"num_beams": 5, "do_sample": False, "length_penalty": 1.0, "max_new_tokens": 150,
               "repetition_penalty": 1.15, "no_repeat_ngram_size": 4, "penalties": "builtin"},
    # Whisper's recommended decoding with temperature fallback (needs a recent transformers).
    "whisper_standard": {"num_beams": 5, "do_sample": False, "max_new_tokens": 440,
                         "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                         "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0,
                         "penalties": "none", "systems": ("whisper",)},
}


def presets_for(system):
    return [name for name, p in DECODING_PRESETS.items() if system in p.get("systems", (system,))]


class GeneratedTextRepetitionPenalty(LogitsProcessor):
    """CTRL-style repetition penalty that only sees tokens generated in this call: never the prompt
    (Dual-Fusion's pad-filled input_ids, Whisper's <|el|><|transcribe|>...) and never special
    tokens, so EOS is not penalized for either system."""

    def __init__(self, penalty, special_ids):
        self.penalty = float(penalty)
        self.special_ids = torch.tensor(sorted(set(special_ids)), dtype=torch.long)
        self.prompt_len = None

    def __call__(self, input_ids, scores):
        if self.prompt_len is None:  # the first call sees exactly the prompt
            self.prompt_len = input_ids.shape[1]
        generated = input_ids[:, self.prompt_len:]
        if generated.shape[1] == 0:
            return scores
        picked = torch.gather(scores, 1, generated)
        picked = torch.where(picked < 0, picked * self.penalty, picked / self.penalty)
        penalized = scores.scatter(1, generated, picked)
        special = self.special_ids.to(scores.device)
        penalized[:, special] = scores[:, special]
        return penalized


class GeneratedTextNoRepeatNGram(LogitsProcessor):
    """no_repeat_ngram_size restricted to generated tokens; the prompt is ignored."""

    def __init__(self, ngram_size):
        self.n = int(ngram_size)
        self.prompt_len = None

    def __call__(self, input_ids, scores):
        if self.prompt_len is None:
            self.prompt_len = input_ids.shape[1]
        n = self.n
        for row, seq in enumerate(input_ids[:, self.prompt_len:].tolist()):
            if len(seq) < n:
                continue
            prefix = seq[len(seq) - n + 1:]
            banned = {seq[i + n - 1] for i in range(len(seq) - n + 1) if seq[i:i + n - 1] == prefix}
            if banned:
                scores[row, list(banned)] = -float("inf")
        return scores


def resolve_decoding(name, system, special_ids, max_new_tokens=None):
    """Return (kwargs for .generate(), factory of fresh logits processors, JSON-able description)."""
    preset = dict(DECODING_PRESETS[name])
    systems = preset.pop("systems", None)
    if systems is not None and system not in systems:
        raise ValueError(f"decoding preset '{name}' is only defined for {systems}")
    mode = preset.pop("penalties")
    if max_new_tokens is not None:
        preset["max_new_tokens"] = int(max_new_tokens)
    description = {"preset": name, "penalties": mode, **preset}
    if mode != "generated_only":
        return preset, LogitsProcessorList, description

    repetition_penalty = preset.pop("repetition_penalty", 1.0)
    ngram_size = preset.pop("no_repeat_ngram_size", 0)
    preset.update(repetition_penalty=1.0, no_repeat_ngram_size=0)  # keep HF built-ins off

    def factory():  # the processors store the prompt length, so build fresh ones per call
        processors = LogitsProcessorList()
        if repetition_penalty > 1.0:
            processors.append(GeneratedTextRepetitionPenalty(repetition_penalty, special_ids))
        if ngram_size > 0:
            processors.append(GeneratedTextNoRepeatNGram(ngram_size))
        return processors

    return preset, factory, description


# ============================================================================ sharded run
def dist_info():
    """(rank, world_size, local_rank) from torchrun's environment; (0, 1, 0) under plain python."""
    return (int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)),
            int(os.environ.get("LOCAL_RANK", 0)))


def base_run_config(system, manifest_path, manifest_meta, decoding, batch_size, feature_extractor, **extra):
    import transformers

    return {"system": system, "manifest": abspath(manifest_path), "manifest_sha256": manifest_meta["sha256"],
            "language": manifest_meta.get("language_code"), "decoding": decoding, "batch_size": batch_size,
            "world_size": dist_info()[1],
            "audio_policy": f"first {MAX_AUDIO_S:.0f} s, log-mel padded to {MAX_AUDIO_S:.0f} s",
            "feature_extractor": {"n_mels": feature_extractor.feature_size,
                                  "sampling_rate": feature_extractor.sampling_rate},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "jiwer": package_version("jiwer")},
            **extra}


def _config_hash(run_config):
    return hashlib.sha256(json.dumps(run_config, sort_keys=True, default=list).encode()).hexdigest()[:16]


def prepare_out_dir(out_dir, overwrite=False):
    """Call before loading any model, on every rank."""
    rank = dist_info()[0]
    if os.path.exists(os.path.join(out_dir, "predictions.jsonl")) and not overwrite:
        raise FileExistsError(f"{out_dir} already holds results; pass --overwrite or choose a new --out")
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    if rank == 0:  # markers from a crashed run; the other ranks only write theirs after decoding
        for name in os.listdir(shard_dir):
            if name.endswith(".done.json"):
                os.remove(os.path.join(shard_dir, name))


def run_sharded(transcriber, rows, manifest_meta, out_dir, batch_size, run_config, poll_s=10, timeout_h=48):
    """Each rank decodes rows[rank::world] and writes a shard; rank 0 waits for all shards,
    checks that every manifest id appears exactly once, then writes and scores the merged run.
    No torch.distributed collective is used, so there is no barrier timeout to hit."""
    from tqdm import tqdm

    rank, world, _ = dist_info()
    shard_dir = os.path.join(out_dir, "shards")
    digest = _config_hash(run_config)
    shard = rows[rank::world]

    records = []
    batches = iter_audio_batches(shard, batch_size)
    for batch_rows, arrays in tqdm(batches, total=math.ceil(len(shard) / batch_size),
                                   desc=f"rank {rank}", position=rank):
        outputs = transcriber.transcribe(batch_rows, arrays)
        if len(outputs) != len(batch_rows):
            raise RuntimeError("the transcriber returned a different number of outputs than inputs")
        records.extend({**row, **output} for row, output in zip(batch_rows, outputs))
    _write_jsonl(os.path.join(shard_dir, f"shard_{rank}.jsonl"), records)
    _write_json(os.path.join(shard_dir, f"shard_{rank}.done.json"),
                {"config": digest, "world": world, "n": len(records)})
    if rank != 0:
        return None

    deadline = time.time() + timeout_h * 3600
    pending = set(range(world))
    while pending:
        for r in sorted(pending):
            marker = os.path.join(shard_dir, f"shard_{r}.done.json")
            if os.path.exists(marker):
                with open(marker, encoding="utf-8") as f:
                    info = json.load(f)
                if info["config"] == digest and info["world"] == world:
                    pending.discard(r)
        if pending:
            if time.time() > deadline:
                raise TimeoutError(f"shards {sorted(pending)} never finished")
            time.sleep(poll_s)

    merged = [rec for r in range(world) for rec in _read_jsonl(os.path.join(shard_dir, f"shard_{r}.jsonl"))]
    expected = [row["id"] for row in rows]
    got = [rec["id"] for rec in merged]
    missing, unexpected = set(expected) - set(got), set(got) - set(expected)
    duplicated = len(got) - len(set(got))
    if missing or unexpected or duplicated:
        raise RuntimeError(f"merge check failed: {len(missing)} missing, {len(unexpected)} unexpected, "
                           f"{duplicated} duplicated ids")
    order = {uid: i for i, uid in enumerate(expected)}
    merged.sort(key=lambda rec: order[rec["id"]])

    _write_jsonl(os.path.join(out_dir, "predictions.jsonl"), merged)
    _write_json(os.path.join(out_dir, "run_config.json"),
                {**run_config, "config_hash": digest, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    metrics = score_records(merged)
    _write_json(os.path.join(out_dir, "metrics.json"), metrics)
    for r in range(world):
        for suffix in (".jsonl", ".done.json"):
            os.remove(os.path.join(shard_dir, f"shard_{r}{suffix}"))
    print_metrics(f"{run_config['system']} [{run_config['decoding']['preset']}] -> {out_dir}", metrics)
    return metrics


# ============================================================================ scoring
SUBSETS = {
    "all": lambda r: True,
    "le_30s": lambda r: r["duration"] <= MAX_AUDIO_S,  # both systems saw the complete audio
    "no_digits": lambda r: re.search(r"\d", r["reference_raw"]) is None,  # no digit/word convention issue
}


def _utterance_counts(refs, hyps, unit="word"):
    """Per utterance: [substitutions, deletions, insertions, reference length]."""
    process = jiwer.process_words if unit == "word" else jiwer.process_characters
    counts = np.zeros((len(refs), 4), dtype=np.int64)
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        if not hyp:
            n = len(ref.split()) if unit == "word" else len(ref)
            counts[i] = (0, n, 0, n)
            continue
        out = process(ref, hyp)
        counts[i] = (out.substitutions, out.deletions, out.insertions,
                     out.hits + out.substitutions + out.deletions)
    return counts


def utterance_table(records):
    refs = [r["reference_raw"] for r in records]
    hyps = [r.get("hypothesis_raw") or "" for r in records]
    n_refs, n_hyps = [normalize_nwer(t) for t in refs], [normalize_nwer(t) for t in hyps]
    p_refs, p_hyps = [normalize_wer_punct(t) for t in refs], [normalize_wer_punct(t) for t in hyps]
    return {"n_wer": _utterance_counts(n_refs, n_hyps),
            "n_cer": _utterance_counts(n_refs, n_hyps, unit="char"),
            "wer_punct": _utterance_counts(p_refs, p_hyps)}


def _rate(counts):
    n_ref = counts[:, 3].sum()
    return float(counts[:, :3].sum() / n_ref) if n_ref else float("nan")


def score_records(records):
    """Corpus-level (micro-averaged) metrics for each subset."""
    table = utterance_table(records)
    metrics = {}
    for name, keep in SUBSETS.items():
        mask = np.array([keep(r) for r in records], dtype=bool)
        words = table["n_wer"][mask]
        metrics[name] = {
            "n_utts": int(mask.sum()), "n_ref_words": int(words[:, 3].sum()),
            "n_wer": _rate(words), "n_cer": _rate(table["n_cer"][mask]),
            "wer_punct": _rate(table["wer_punct"][mask]),
            "sub": int(words[:, 0].sum()), "del": int(words[:, 1].sum()), "ins": int(words[:, 2].sum()),
            "cap_hits": sum(bool(r.get("hit_cap")) for r, m in zip(records, mask) if m),
        }
    return metrics


def print_metrics(label, metrics):
    print(f"\n{label}")
    print(f"{'subset':<10}{'utts':>7}{'N-WER':>8}{'N-CER':>8}{'WER(p)':>8}{'sub':>7}{'del':>7}{'ins':>7}{'cap':>6}")
    for name, m in metrics.items():
        print(f"{name:<10}{m['n_utts']:>7}{100 * m['n_wer']:>8.2f}{100 * m['n_cer']:>8.2f}"
              f"{100 * m['wer_punct']:>8.2f}{m['sub']:>7}{m['del']:>7}{m['ins']:>7}{m['cap_hits']:>6}")


# ============================================================================ comparison
def load_run(run_dir):
    with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    return config, _read_jsonl(os.path.join(run_dir, "predictions.jsonl"))


def paired_bootstrap(errors_a, errors_b, n_ref, n_boot=2000, seed=0, chunk=250):
    """Resample utterances; return (observed delta, 95% CI low, high, two-sided p) of corpus WER A-B."""
    rng = np.random.default_rng(seed)
    n = len(n_ref)
    deltas = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        deltas.append((errors_a[idx].sum(axis=1) - errors_b[idx].sum(axis=1)) / n_ref[idx].sum(axis=1))
    deltas = np.concatenate(deltas)
    observed = (errors_a.sum() - errors_b.sum()) / n_ref.sum()
    low, high = np.percentile(deltas, [2.5, 97.5])
    p_value = min(1.0, 2.0 * min((deltas >= 0).mean(), (deltas <= 0).mean()))
    return float(observed), float(low), float(high), float(p_value)


def compare_runs(dir_a, dir_b, n_boot=2000, seed=0):
    """Re-score both runs with the current normalizer and test A-B on identical utterances."""
    config_a, records_a = load_run(dir_a)
    config_b, records_b = load_run(dir_b)
    if config_a["manifest_sha256"] != config_b["manifest_sha256"]:
        raise RuntimeError("the two runs used different test manifests")
    by_id = {r["id"]: r for r in records_b}
    if {r["id"] for r in records_a} != set(by_id) or len(by_id) != len(records_b):
        raise RuntimeError("the two runs do not cover exactly the same utterances")
    records_b = [by_id[r["id"]] for r in records_a]
    if any(a["reference_raw"] != b["reference_raw"] for a, b in zip(records_a, records_b)):
        raise RuntimeError("reference text differs between the runs")

    label_a = f"{config_a['system']} [{config_a['decoding']['preset']}]"
    label_b = f"{config_b['system']} [{config_b['decoding']['preset']}]"
    if config_a["decoding"] != config_b["decoding"]:
        print("NOTE: decoding differs between the runs, so this is not the controlled comparison.")
        print(f"  A: {config_a['decoding']}\n  B: {config_b['decoding']}")
    table_a, table_b = utterance_table(records_a), utterance_table(records_b)

    print(f"\nA = {label_a}\nB = {label_b}\n")
    print(f"{'subset':<10}{'metric':<10}{'utts':>6}{'A':>8}{'B':>8}{'A-B':>8}{'95% CI':>18}{'p':>8}")
    for name, keep in SUBSETS.items():
        mask = np.array([keep(r) for r in records_a], dtype=bool)
        if not mask.any():
            continue
        for key, title in (("n_wer", "N-WER"), ("wer_punct", "WER(p)")):
            a, b = table_a[key][mask], table_b[key][mask]
            delta, low, high, p_value = paired_bootstrap(a[:, :3].sum(1), b[:, :3].sum(1), a[:, 3], n_boot, seed)
            ci = f"[{100 * low:+.2f}, {100 * high:+.2f}]"
            print(f"{name:<10}{title:<10}{int(mask.sum()):>6}{100 * _rate(a):>8.2f}{100 * _rate(b):>8.2f}"
                  f"{100 * delta:>+8.2f}{ci:>18}{p_value:>8.3f}")
    cap_a = sum(bool(r.get("hit_cap")) for r in records_a)
    cap_b = sum(bool(r.get("hit_cap")) for r in records_b)
    print(f"\nOutputs that hit the token cap: A={cap_a}, B={cap_b} (should be 0 in the controlled comparison)")


# ============================================================================ CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest", help="freeze the test set shared by all systems")
    build.add_argument("--exp", type=int, required=True)
    build.add_argument("--datasets", nargs="+", required=True)
    build.add_argument("--split", default="test")
    build.add_argument("--out", required=True)
    build.add_argument("--skip_train_overlap", action="store_true")

    score = sub.add_parser("score", help="re-score a finished run with the current normalizer")
    score.add_argument("run_dir")

    compare = sub.add_parser("compare", help="paired bootstrap between two runs on the same manifest")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.add_argument("--n_boot", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "build-manifest":
        meta = build_manifest(args.exp, args.datasets, args.split, args.out, not args.skip_train_overlap)
        print(json.dumps({k: v for k, v in meta.items() if not k.endswith("_ids")}, indent=2, ensure_ascii=False))
    elif args.command == "score":
        config, records = load_run(args.run_dir)
        metrics = score_records(records)
        _write_json(os.path.join(args.run_dir, "metrics.json"), metrics)
        print_metrics(f"{config['system']} [{config['decoding']['preset']}]", metrics)
    else:
        compare_runs(args.run_a, args.run_b, args.n_boot, args.seed)


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------------------------------------
# SPEECHLM UTILS
# -------------------------------------------------------------------------------------------------------------------

import argparse
import json
import os
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from speechLM_utils.environment import set_environment

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
set_environment()

import safetensors.torch
import torch
import transformers
from transformers import GenerationConfig

transformers.logging.set_verbosity_error()

from evaluations import fair_eval as fe
from speechLM_utils.data_collator import DATASET_PROMPT_MAP, PROMPT_FORMAL, PROMPT_VERBATIM
from speechLM_utils.model import get_max_step, get_model
from speechLM_utils.utils import init, init_gpu

# Parameters that were trained (bridge + LoRA) and therefore must come from the checkpoint.
TRAINED_PARAM_MARKERS = ("lora_", "adapter.", "input_downsampler", "injection_downsampler",
                         "cross_attention_layer", "layer_dynamic", "layer_static")
# bitsandbytes quantization state of the frozen base LLM; reloaded from the Hub, never needed.
QUANT_STATE_MARKERS = ("absmax", "quant_map", "quant_state", "bitsandbytes", "nested_")


# ============================================================================ checkpoint
def resolve_checkpoint(checkpoint_path, explicit, select):
    if explicit:
        return explicit
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

# -------------------------------------------------------------------------------------------------------------------
# LOADING MODEL
# -------------------------------------------------------------------------------------------------------------------

from typing import Dict

from speechLM_utils.dual_fusion_model import DualFusionModel
import torch
from speechLM_utils.checkpoint import get_max_step
import os
import torch.nn as nn
from os.path import join
import json
import safetensors

def load_weights(model: nn.Module, checkpoint_path: str, args: Dict, device: str = 'cuda'):
    max_step = get_max_step(checkpoint_path)
    weights_checkpoint = join(checkpoint_path, f"checkpoint-{max_step}")
    print('Found checkpoint to load: ', weights_checkpoint)

    index_file = os.path.join(weights_checkpoint, "model.safetensors.index.json")
    state_dict = {}

    try:
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            for shard in shard_files:
                shard_path = os.path.join(weights_checkpoint, shard)
                state_dict.update(safetensors.torch.load_file(shard_path))
        elif os.path.exists(os.path.join(weights_checkpoint, "model.safetensors")):
            state_dict = safetensors.torch.load_file(os.path.join(weights_checkpoint, "model.safetensors"))
        elif os.path.exists(os.path.join(weights_checkpoint, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(weights_checkpoint, "pytorch_model.bin"), map_location="cpu")
        else:
            raise FileNotFoundError(f"No valid weights found in {weights_checkpoint}")

        trained_keys = [
            'adapter', 'input_downsampler', 'injection_downsampler', 'injection_downsamplers',
            'cross_attention_layer', 'ctc_predictor', 'audio_predictor', 'duration_predictor'
        ]

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if any(trained_key in k for trained_key in trained_keys):
                new_k = k
                if 'language_model.model.layers' in k:
                    new_k = k.replace('language_model.model.layers', 'language_model.base_model.model.model.layers')
                filtered_state_dict[new_k] = v

        if not filtered_state_dict:
            print(f"[{device}] WARNING: No projector/adapter keys found in Stage 1 checkpoint!")

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected_keys)

        missing_cross_attn = [k for k in missing_keys if 'cross_attention' in k]
        if missing_cross_attn:
            print(f"[{device}] CRITICAL WARNING: Failed to load cross-attention weights!")

        print(f"[{device}] SUCCESS: Stage 1 weights loaded.")
        print(f"[{device}] Loaded {len(filtered_state_dict)} tensors.")

    except Exception as e:
        print(f"[{device}] Error: Could not load weights. {e}")

    return model, max_step

def get_model(args: Dict, info: Dict, device = 'cuda', exp: int = 0, attn_implementation: str = 'sdpa'):

    model = DualFusionModel(
        speech_encoder_model_id=info['speech_encoder_id'],
        language_model_id=info['language_model_id'],
        blank_training=args['blank_training'],
        audio_dropout=args['audio_dropout'],
        text_perturbation=args['text_perturbation'],
        text_dropout=args['text_dropout'],
        spec_augment=args['spec_augment'],
        include_adapter=args['include_adapter'],
        input_downsample=args['input_downsample'],
        static_projector=args['static_projector'],
        downsample_K=args['downsample_K'],
        hidden_dim=args['hidden_dim'],
        static_injection=args['static_injection_layers'],
        injection_layers=args['injection_layers'],
        pyramid_layers=args['pyramid_layers'],
        gated=args['gated_cross_attention'],
        downsample_L=args['downsample_L'],
        injection_downsample=args['injection_downsample'],
        downsamplers=args['downsamplers'],
        causal_fusion=args['causal_fusion'],
        positional_info=args['positional_info'],
        layer_wise_fusion=args['layer_wise_fusion'],
        layer_weights_static=args['layer_weights_static'],
        predict_duration=args['predict_duration'],
        duration_resolution=args['duration_resolution'],
        max_duration=args['max_duration'],
        ctc=args['ctc'],
        audio_forecasting=args['audio_forecasting'],
        ctc_weight=args['ctc_weight'],
        audio_weight=args['audio_weight'],
        duration_weight=args['duration_weight'],
        lng_lora=args['linguistic_lora'],
        lora_r=args['lora_r'],
        acoustic_lora=args['acoustic_lora'],
        lora_params=args['lora_params'],
        prompt_persona=args['prompt_persona'],
        prompt_instruction=args['prompt_instruction'],
        prompt_verbatim=args['prompt_verbatim'],
        dtype=torch.float32,
        bit4=True,
        attn_implementation=attn_implementation,
        device=device,
        exp=exp
    )

    tokenizer = model.language_tokenizer

    return model, tokenizer

# -------------------------------------------------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------------------------------------------------

import gc
import torch
import os
import wandb
from config.parser import Parser
from speechLM_utils.checkpoint import get_checkpoint
import json
from transformers import GenerationConfig, Seq2SeqTrainingArguments
from typing import Dict, List
import socket

def tags_to_str(tags: List[str]):
    return '\n'.join(tags)

def get_tags(args: Dict, info: Dict, datasets: List[str]):
    tags = []
    tags.append(socket.gethostname())

    tags.append(f"Speech Encoder: {info['speech_encoder_id']}")
    tags.append(f"Language Model: {info['language_model_id']}")
    tags.append(f"Interleave Factor: {info['interleave']}")

    ds_string = ','.join(datasets)
    if len(ds_string) > 64:
        tags.append(", ".join([dataset[:5] for dataset in datasets]))
    else:
        tags.append(ds_string)

    ## Regularization & Augmentation Configurations
    if args['text_perturbation']:
        tags.append('text_perturbation')
        tags.append('text_dropout: ' + str(args['text_dropout']))

    ## Input Injection Configurations
    if args['include_adapter']:
        tags.append('include_adapter')
        tags.append('downsample_K: ' + str(args['downsample_K']))

    ## Cross-Attention Injection Configurations
    tags.append('downsample_L: ' + str(args['downsample_L']))
    tags.append('injection_layers: ' + ', '.join(str(layer) for layer in args['injection_layers']))
    if args['gated_cross_attention']:
        tags.append('gated_cross_attention')
    if args['causal_fusion']:
        tags.append('causal_fusion')
    if args['positional_info']:
        tags.append('positional_info')
    if args['pyramid_layers']:
        tags.append('pyramid_layers')
    if args['layer_wise_fusion']:
        tags.append('layer_wise_fusion')

    ## LoRA Configurations
    if args['linguistic_lora']:
        tags.append('linguistic_lora')
        tags.append('lora_r: ' + str(args['lora_r']))
    if args['two_stage']:
        tags.append('two_stage')

    ## Auxiliary Losses Configurations
    if args['predict_duration']:
        tags.append('predict_duration')
    if args['ctc']:
        tags.append('ctc')
    if args['audio_forecasting']:
        tags.append('audio_forecasting')

    ## Prompt Configurations
    if args['prompt_persona'] is not None:
        persona_len = len(args['prompt_persona'])
        tags.append(f'prompt_persona len: {persona_len}')
    if args['prompt_instruction'] is not None:
        instruct_len = len(args['prompt_instruction'])
        tags.append(f'prompt_instruction: {instruct_len}')

    if args['prompt_verbatim']:
        tags.append('prompt_verbatim')

    return tags


def init_gpu():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"Process launched on GPU: {local_rank}")
    else:
        device = "cuda"
        print("Running in standard mode (DataParallel or Single GPU)")

    return local_rank, device


def init_info(speech_encoder_id, language_model_id, datasets, machine=None, datetime=None, interleave=True, iters=800):
    model_name = (speech_encoder_id.split('/')[1] + '_' + language_model_id.split('/')[1])

    info = {
        'checkpoint_folder': 'dual_fusion_checkpoints',
        'model_name': model_name,
        'train_dataset': datasets,
        'interleave': interleave,
        'speech_encoder_id': speech_encoder_id,
        'language_model_id': language_model_id,
        'machine': machine,
        'datetime': datetime,
        'iters': iters,
        'do_compute': True,
        'compute_wer_cer': True,
        'N_samples_for_metrics': None

    }

    return info


def init(info, restart=True, exp: int = 0):
    conf = Parser()
    conf.get_args(exp)

    args = conf.dual_fuse_args
    training_args = args.training_args

    checkpoint_path, checkpoint_dir, writer, date, loaded_args = get_checkpoint(args.checkpoint_path,
                                                                                info,
                                                                                restart)

    if loaded_args is not None:
        args.__dict__.update(loaded_args)
        training_args = args.training_args

    else:
        config_file = os.path.join(checkpoint_path, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(args.__dict__, f)

    training_args['output_dir'] = checkpoint_path
    training_args['logging_dir'] = checkpoint_path
    training_args['report_to'] = ['wandb']
    training_args['generation_config'] = GenerationConfig(**training_args['generation_config'])
    training_args = Seq2SeqTrainingArguments(**training_args)

    return conf, args, training_args, date, checkpoint_path, checkpoint_dir, writer


def resume_wandb(local_rank, date, model_name, exp):
    id = f'{date}-Exp:{exp}'
    if local_rank in [-1, 0]:
        api = wandb.Api()
        runs = api.runs("chrissiargas-innoetics/Hungarian-ASR",
                        filters={"display_name": id})

        if len(runs) > 0:
            run_id = runs[0].id
            print(f"Found existing W&B run '{id}' with ID {run_id}. Resuming...")
            wandb.init(project="Hungarian-ASR",
                       entity="chrissiargas-innoetics",
                       id=run_id, resume="must")
        else:
            print(f"Could not find existing run '{id}'. Starting a new evaluation run...")
            wandb.init(project="Hungarian-ASR",
                       entity="chrissiargas-innoetics",
                       name=f"{id}_Eval",
                       group=model_name)


def init_wandb(local_rank, args, info, date, model_name, datasets, exp):
    tags = get_tags(args, info, datasets)
    note = tags_to_str(tags)

    if local_rank in [-1, 0]:
        wandb.init(
            entity="chrissiargas-innoetics",
            project="Hungarian-ASR",
            name=f'{date}-Exp:{exp}',
            group=model_name,
            config=args.__dict__,
            tags=tags,
            notes=note,
            reinit=True
        )