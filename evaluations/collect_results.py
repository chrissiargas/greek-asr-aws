"""Collect the finished runs into one table: one row per dataset, N-WER / WER_p / N-CER for
the Dual-Fusion model and for Whisper.

  python evaluations/collect_results.py runs/hungarian_*_test
  python evaluations/collect_results.py runs/greek_*_test --model dual_fusion_plain \
      --whisper whisper_plain --csv results/greek.csv --latex results/greek.tex
  python evaluations/collect_results.py runs/hungarian_*_test --bootstrap   # adds 95% CI and p

Each argument is a test folder holding one sub-folder per system (e.g. dual_fusion_plain,
whisper_plain), each with the metrics.json written by fair_eval. The system folders are found
automatically unless --model/--whisper name them; the dataset name comes from the folder name
(hungarian_fleurs_test -> fleurs).
"""
import argparse
import json
import os
import sys
from os.path import basename, dirname, isdir, join

sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

METRIC_KEYS = (("n_wer", "N-WER"), ("wer_punct", "WER_p"), ("n_cer", "N-CER"))


def dataset_name(run_dir):
    name = basename(run_dir.rstrip("/"))
    for prefix in ("hungarian_", "greek_", "hu_", "el_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name[:-5] if name.endswith("_test") else name


def find_system(run_dir, prefix, explicit=None):
    """Return the sub-folder for one system, or None when the run is missing."""
    if explicit:
        path = join(run_dir, explicit)
        return path if os.path.exists(join(path, "metrics.json")) else None
    matches = [join(run_dir, d) for d in sorted(os.listdir(run_dir))
               if d.startswith(prefix) and isdir(join(run_dir, d))
               and os.path.exists(join(run_dir, d, "metrics.json"))]
    if len(matches) > 1:
        raise SystemExit(f"{run_dir} holds several '{prefix}*' runs ({', '.join(map(basename, matches))}); "
                         f"choose one with --model/--whisper")
    return matches[0] if matches else None


def load(system_dir, subset):
    if system_dir is None:
        return None
    with open(join(system_dir, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)[subset]
    config = {}
    if os.path.exists(join(system_dir, "run_config.json")):
        with open(join(system_dir, "run_config.json"), encoding="utf-8") as f:
            config = json.load(f)
    return {"dir": system_dir, "metrics": metrics, "config": config}


def bootstrap(model, whisper, n_boot, seed):
    """Paired bootstrap of the N-WER difference, recomputed from the stored predictions."""
    from evaluations import fair_eval as fe

    def rows(run):
        path = join(run["dir"], "predictions.jsonl")
        return [json.loads(line) for line in open(path, encoding="utf-8")]

    a, b = rows(model), rows(whisper)
    by_id = {r["id"]: r for r in b}
    if {r["id"] for r in a} != set(by_id):
        return None
    b = [by_id[r["id"]] for r in a]

    def table(records, config):
        # the Hungarian harness scores each run with its own number-verbalization setting
        verbalizes = getattr(fe, "run_verbalizes", None)
        return fe.utterance_table(records) if verbalizes is None else fe.utterance_table(records, verbalizes(config))

    ta = table(a, model["config"])["n_wer"]
    tb = table(b, whisper["config"])["n_wer"]
    errors_a, errors_b = ta[:, :3].sum(1), tb[:, :3].sum(1)
    try:  # newer signature carries B's own reference lengths
        delta, low, high, p_value = fe.paired_bootstrap(errors_a, errors_b, ta[:, 3], tb[:, 3], n_boot, seed)
    except TypeError:
        delta, low, high, p_value = fe.paired_bootstrap(errors_a, errors_b, ta[:, 3], n_boot, seed)
    return {"delta": 100 * delta, "low": 100 * low, "high": 100 * high, "p": p_value}


def collect(run_dirs, args):
    table = []
    for run_dir in run_dirs:
        if not isdir(run_dir):
            print(f"skipping {run_dir}: not a folder", file=sys.stderr)
            continue
        model = load(find_system(run_dir, "dual_fusion", args.model), args.subset)
        whisper = load(find_system(run_dir, "whisper", args.whisper), args.subset)
        if model is None and whisper is None:
            print(f"skipping {run_dir}: no finished run", file=sys.stderr)
            continue
        row = {"dataset": dataset_name(run_dir),
               "n_utts": (model or whisper)["metrics"]["n_utts"],
               "ours": model["metrics"] if model else None,
               "whisper": whisper["metrics"] if whisper else None,
               "presets": [(r["config"].get("decoding", {}) or {}).get("preset", "?") if r else "-"
                           for r in (model, whisper)]}
        if args.bootstrap and model and whisper:
            row["boot"] = bootstrap(model, whisper, args.n_boot, args.seed)
        table.append(row)
    return sorted(table, key=lambda r: r["dataset"])


def cell(metrics, key):
    return "-" if metrics is None else f"{100 * metrics[key]:.1f}"


def print_table(table, args):
    presets = {tuple(r["presets"]) for r in table}
    print(f"\nsubset: {args.subset}   decoding (ours, whisper): "
          f"{'; '.join('/'.join(p) for p in sorted(presets))}\n")
    head = f"{'dataset':<18}{'utts':>6}" + "".join(f"{f'{t} {s}':>13}" for _, t in METRIC_KEYS for s in ("ours", "wh."))
    if args.bootstrap:
        head += f"{'Δ N-WER':>10}{'95% CI':>18}{'p':>8}"
    print(head)
    for row in table:
        line = f"{row['dataset']:<18}{row['n_utts']:>6}"
        for key, _ in METRIC_KEYS:
            line += f"{cell(row['ours'], key):>13}{cell(row['whisper'], key):>13}"
        boot = row.get("boot")
        if args.bootstrap:
            if boot:
                ci = "[{:+.2f}, {:+.2f}]".format(boot["low"], boot["high"])
                line += "{:>+10.2f}{:>18}{:>8.3f}".format(boot["delta"], ci, boot["p"])
            else:
                line += "{:>10}{:>18}{:>8}".format("-", "-", "-")
        print(line)


def write_csv(table, path, args):
    import csv

    fields = ["dataset", "n_utts"] + [f"{s}_{k}" for k, _ in METRIC_KEYS for s in ("ours", "whisper")]
    if args.bootstrap:
        fields += ["delta_n_wer", "ci_low", "ci_high", "p_value"]
    os.makedirs(dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in table:
            out = {"dataset": row["dataset"], "n_utts": row["n_utts"]}
            for key, _ in METRIC_KEYS:
                out[f"ours_{key}"] = cell(row["ours"], key)
                out[f"whisper_{key}"] = cell(row["whisper"], key)
            if args.bootstrap and row.get("boot"):
                out.update(delta_n_wer=f"{row['boot']['delta']:.2f}", ci_low=f"{row['boot']['low']:.2f}",
                           ci_high=f"{row['boot']['high']:.2f}", p_value=f"{row['boot']['p']:.3f}")
            writer.writerow(out)
    print(f"\nwrote {path}")


def write_latex(table, path, args):
    """booktabs table in the paper's style; the better N-WER of each row is bold."""
    columns = "l | c c | c c | c c" + (" | c" if args.bootstrap else "")
    group = "& \\multicolumn{2}{c|}{\\textbf{N-WER (\\%)}} & \\multicolumn{2}{c|}{\\textbf{WER$_p$ (\\%)}} " \
            "& \\multicolumn{2}{c}{\\textbf{N-CER (\\%)}}"
    header = "\\textbf{Dataset} & \\textbf{Ours} & \\textbf{Whisper} & \\textbf{Ours} & \\textbf{Whisper} " \
             "& \\textbf{Ours} & \\textbf{Whisper}"
    if args.bootstrap:
        group += " & \\textbf{$\\Delta$ N-WER}"
        header += " & \\textbf{[95\\% CI]}"
    lines = ["\\begin{table}[h]", "\\centering",
             "\\caption{ASR performance. N-WER is the primary metric; WER$_p$ counts punctuation "
             "as tokens. Both systems use the same test utterances, input and decoding.}",
             "\\label{tab:results}", "\\footnotesize", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{" + columns + "}", "\\toprule",
             group + " \\\\",
             "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
             header + " \\\\", "\\midrule"]
    for row in table:
        cells = []
        for key, _ in METRIC_KEYS:
            ours, whisper = cell(row["ours"], key), cell(row["whisper"], key)
            if key == "n_wer" and "-" not in (ours, whisper):
                better = min((float(ours), "ours"), (float(whisper), "whisper"))[1]
                ours = f"\\textbf{{{ours}}}" if better == "ours" else ours
                whisper = f"\\textbf{{{whisper}}}" if better == "whisper" else whisper
            cells += [ours, whisper]
        if args.bootstrap:
            boot = row.get("boot")
            cells.append(f"{boot['delta']:+.1f} [{boot['low']:+.1f}, {boot['high']:+.1f}]" if boot else "-")
        lines.append(f"{row['dataset'].replace('_', ' ').title()} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    os.makedirs(dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="test folders, e.g. runs/hungarian_*_test")
    parser.add_argument("--model", default=None, help="model sub-folder name (default: the dual_fusion* run)")
    parser.add_argument("--whisper", default=None, help="Whisper sub-folder name (default: the whisper* run)")
    parser.add_argument("--subset", default="all", choices=["all", "le_30s", "no_digits"])
    parser.add_argument("--bootstrap", action="store_true", help="add the paired-bootstrap CI and p-value")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--latex", default=None)
    args = parser.parse_args()

    table = collect(args.run_dirs, args)
    if not table:
        raise SystemExit("no finished runs found")
    print_table(table, args)
    if args.csv:
        write_csv(table, args.csv, args)
    if args.latex:
        write_latex(table, args.latex, args)


if __name__ == "__main__":
    main()
