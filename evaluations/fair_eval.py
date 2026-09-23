
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
