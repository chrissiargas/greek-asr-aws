from preprocessing.normalize import normalize
import jiwer
import pandas as pd
from typing import Tuple


def get_metrics(predictions, references, indices, durations, verbose: bool = True) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    norm_predictions = [normalize(t, with_signs=False) for t in predictions]
    norm_references = [normalize(t, with_signs=False) for t in references]

    predictions = [normalize(t, with_signs=True) for t in predictions]
    references = [normalize(t, with_signs=True) for t in references]

    valid_data = [
        (p, r, np, nr, i, d) for p, r, np, nr, i, d in zip(
            predictions, references, norm_predictions, norm_references, indices, durations
        ) if len(r.strip()) > 0 and len(nr.strip()) > 0
    ]

    dropped_count = len(references) - len(valid_data)
    if dropped_count > 0 and verbose:
        print(f"Skipping {dropped_count} samples due to empty reference strings after normalization.")

    if not valid_data:
        return pd.DataFrame(), pd.DataFrame()

    predictions, references, norm_predictions, norm_references, indices, durations = map(list, zip(*valid_data))

    wers, cers, ratio = [], [], []
    n_wers, n_cers, n_ratio = [], [], []
    subs, ins, dels = [], [], []

    for reference, prediction, n_reference, n_prediction in zip(references, predictions, norm_references, norm_predictions):
        wers.append(jiwer.wer(reference, prediction))
        n_wers.append(jiwer.wer(n_reference, n_prediction))
        cers.append(jiwer.cer(reference, prediction))
        n_cers.append(jiwer.cer(n_reference, n_prediction))
        ratio.append(len(prediction) / len(reference) if len(reference) > 0 else 0)
        n_ratio.append(len(n_prediction) / len(n_reference) if len(reference) > 0 else 0)

        out = jiwer.process_words(reference, prediction)
        subs.append(out.substitutions)
        ins.append(out.insertions)
        dels.append(out.deletions)

    results_per_instance = {'index': indices,
                            'reference': references,
                            'prediction': predictions,
                            'wer': wers,
                            'n_wer': n_wers,
                            'cer': cers,
                            'n_cer': n_cers,
                            'ratio': ratio,
                            'n_ratio': n_ratio,
                            'substitutions': subs,
                            'insertions': ins,
                            'deletions': dels,
                            'duration': durations}

    results_per_instance = pd.DataFrame(results_per_instance)

    if verbose:
        total_wer = jiwer.wer(references, predictions)
        total_n_wer = jiwer.wer(norm_references, norm_predictions)
        total_cer = jiwer.cer(references, predictions)
        total_n_cer = jiwer.cer(norm_references, norm_predictions)
        total_out = jiwer.process_words(references, predictions)

        print("-" * 50)
        print(f"Global Metrics (Micro-Average over {len(predictions)} samples):")
        print(f"Total WER  : {total_wer:.4f}")
        print(f"Total N_WER: {total_n_wer:.4f}")
        print(f"Total CER  : {total_cer:.4f}")
        print(f"Total N_CER: {total_n_cer:.4f}")
        print(f"Total Substitutions: {total_out.substitutions:.4f}")
        print(f"Total Insertions: {total_out.insertions:.4f}")
        print(f"Total Deletions: {total_out.deletions:.4f}")
        print("-" * 50)

        total_results = {
            'wer': [total_wer],
            'n_wer': [total_n_wer],
            'cer': [total_cer],
            'n_cer': [total_n_cer],
            'subs': [total_out.substitutions],
            'ins': [total_out.insertions],
            'dels': [total_out.deletions]
        }

        total_results = pd.DataFrame(total_results)
    else:
        total_results = None

    return results_per_instance, total_results