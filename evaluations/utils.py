import os

import safetensors.torch
import torch
from os.path import dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from config.parser import Parser
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import argparse
from bert_score import score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.ndimage as ndimage

def get_results_path(conf, args, data_specific=True):

    model_type = args.get('model_type', 'dual_fusion_checkpoints')
    model_name = args.get('model_name', 'default_model')

    path_parts = [
        os.path.expanduser('~'),
        conf.results_path,
        model_type
    ]

    if args.get('machine') and args.get('datetime') and args.get('turn'):
        path_parts.extend([
            str(args['machine']),
            str(model_name),
            str(args['datetime']),
            str(args['turn'])
        ])
    else:
        path_parts.append(model_name)

    dataset = args.get('dataset')

    if data_specific and dataset:
        path_parts.extend([
            dataset,
            args.get('split', 'test')
        ])

        if args.get('name'):
            path_parts.append(args['name'])

    results_path = os.path.join(*[p for p in path_parts if p])
    os.makedirs(results_path, exist_ok=True)
    return results_path


def get_plots_dir(conf, info):
    base_res_dir = get_results_path(conf, info)
    plots_dir = os.path.join(base_res_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    return plots_dir


def generate_semantic_drift_plot(info: Dict):
    conf = Parser()
    conf.get_args()
    local_info = info.copy()

    for dataset_name in DATASETS:
        local_info['test_dataset'] = dataset_name
        results_folder = get_results_path(conf, local_info)
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if os.path.exists(predictions_path):
            df = pd.read_csv(predictions_path)

        df['reference'] = df['reference'].fillna('')
        df['prediction'] = df['prediction'].fillna('')

        refs = df['reference'].tolist()
        preds = df['prediction'].tolist()

        print("Calculating BERTScore on GPU (using multilingual model for Greek)...")
        P, R, F1 = score(
            preds,
            refs,
            lang="el",
            model_type="bert-base-multilingual-cased",
            batch_size=32,
            verbose=False
        )

        df['semantic_f1'] = F1.tolist()

        plot_df = df[df['wer'] <= 10.0].copy()

        plt.figure(figsize=(11, 8))

        sns.scatterplot(
            data=plot_df,
            x='wer',
            y='semantic_f1',
            palette={True: '#e74c3c', False: '#3498db'},  # Red for loops, Blue for healthy text
            alpha=0.7,
            edgecolor='w',
            s=70
        )

        # Draw quadrant guidelines
        plt.axhline(y=0.80, color='#2ecc71', linestyle='--', alpha=0.8, label='Acceptable Semantics')
        plt.axvline(x=0.20, color='#f39c12', linestyle='--', alpha=0.8, label='Acceptable WER')

        # Annotate the specific regions to explain the "Lazy Decoder" problem
        plt.text(0.02, 0.95, 'Top-Left\nPerfect', color='green', fontsize=10, alpha=0.8)
        plt.text(1.2, 0.95, 'Top-Right\nParaphrasing / Grammar Fixes\n(Lazy Decoder)', color='orange', fontsize=10,
                 alpha=0.8)
        plt.text(1.2, 0.40, 'Bottom-Right\nCatastrophic Hallucination\n(Lost Meaning)', color='red', fontsize=10,
                 alpha=0.8)

        # Formatting
        plt.title(f"Semantic Drift Analysis - {dataset_name}\n(Whisper Acoustics vs KriKri Semantics)", fontsize=14,
                  pad=15)
        plt.xlabel("Word Error Rate (WER) ➔ Lower is Better", fontsize=12)
        plt.ylabel("Semantic F1 (BERTScore) ➔ Higher is Better", fontsize=12)

        # Customize Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out the seaborn legend artifacts
        valid_handles = [h for h, l in zip(handles, labels) if
                         l in ['False', 'True', 'Acceptable Semantics', 'Acceptable WER']]
        valid_labels = [l for l in labels if l in ['False', 'True', 'Acceptable Semantics', 'Acceptable WER']]
        plt.legend(valid_handles, valid_labels, title="Caught in Generation Loop", loc="lower left")

        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        # 5. Save the plot
        save_dir = get_plots_dir(conf, info, 'semantic_drift')
        save_path = os.path.join(save_dir, f"{dataset_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved successfully to: {save_path}\n")


def aggregate_results(info: Dict):
    conf = Parser()
    conf.get_args()
    print("Aggregating results...")

    all_results = []
    local_info = info.copy()

    for dataset_name in DATASETS:
        local_info['dataset'] = dataset_name
        local_info['test_dataset'] = dataset_name
        results_folder = get_results_path(conf, local_info, data_specific=True)

        local_info['res_folder'] = results_folder

        results_path = os.path.join(results_folder, 'predictions_total_metrics.csv')
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if os.path.exists(results_path):
            try:
                df = pd.read_csv(results_path, index_col=0)
                row_data = df.iloc[0].to_dict()
                row_data['dataset'] = dataset_name

                # Calculate Orthographic Gap (OG)
                row_data['og'] = row_data['wer'] - row_data['n_wer']

                # Calculate Catastrophic Rate (CR)
                pred_df = pd.read_csv(predictions_path)
                row_data['catastrophic_rate'] = (pred_df['wer'] > 0.5).mean()

                all_results.append(row_data)
            except Exception as e:
                print(f"Error reading {results_path}: {e}")
        else:
            print(f"Warning: Could not find results for {dataset_name} at {results_path}")

    if not all_results:
        return pd.DataFrame()

    aggregated_df = pd.DataFrame(all_results)

    # Convert to percentages for readability
    perc_cols = ['wer', 'n_wer', 'cer', 'n_cer', 'og', 'catastrophic_rate']
    for col in perc_cols:
        if col in aggregated_df.columns:
            aggregated_df[col] = (aggregated_df[col] * 100).round(1)

    cols = ['dataset'] + [col for col in aggregated_df.columns if col != 'dataset']
    aggregated_df = aggregated_df[cols]

    results_folder = get_results_path(conf, info, data_specific=False)
    aggregated_path = os.path.join(results_folder, "aggregated_results.csv")
    aggregated_df.to_csv(aggregated_path, index=False)
    print(f"✅ Summary compilation saved to: {aggregated_path}")

    return aggregated_df


def compare_models(df_model1, df_model2, name_model1="Model_1", name_model2="Model_2"):
    if df_model1.empty or df_model2.empty:
        return pd.DataFrame()

    comparison_df = pd.merge(
        df_model1,
        df_model2,
        on='dataset',
        suffixes=(f'_{name_model1}', f'_{name_model2}')
    )

    metrics = ['wer', 'n_wer', 'cer', 'n_cer']

    for metric in metrics:
        col1 = f'{metric}_{name_model1}'
        col2 = f'{metric}_{name_model2}'

        if col1 in comparison_df.columns and col2 in comparison_df.columns:
            diff_col = f'{metric}_diff'
            comparison_df[diff_col] = (comparison_df[col2] - comparison_df[col1]).round(2)

            # Relative Error Reduction (RER)
            rer_col = f'{metric}_rer'
            comparison_df[rer_col] = (
                    (comparison_df[col1] - comparison_df[col2]) / (comparison_df[col1] + 1e-9) * 100).round(1)

    return comparison_df


# --- Visualization Functions ---
def plot_sid_stacked_bar(info: Dict):
    conf = Parser()
    conf.get_args()
    sid_data = []
    local_info = info.copy()

    for dataset_name in DATASETS:
        local_info['dataset'] = dataset_name
        results_folder = get_results_path(conf, local_info)
        predictions_path = os.path.join(results_folder, 'predictions.csv')

        if os.path.exists(predictions_path):
            df = pd.read_csv(predictions_path)
            total_subs = df['substitutions'].sum()
            total_ins = df['insertions'].sum()
            total_dels = df['deletions'].sum()
            total_errors = total_subs + total_ins + total_dels

            if total_errors > 0:
                sid_data.append({
                    'Dataset': dataset_name,
                    'Substitutions': (total_subs / total_errors) * 100,
                    'Insertions': (total_ins / total_errors) * 100,
                    'Deletions': (total_dels / total_errors) * 100
                })

    if not sid_data:
        print("No S-I-D data found to plot.")
        return

    sid_df = pd.DataFrame(sid_data).set_index('Dataset')

    ax = sid_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#ffb347', '#ff6961', '#aec6cf'], edgecolor='black')

    plt.title("Proportion of Error Types per Dataset (S-I-D)", fontsize=14, pad=15)
    plt.ylabel("Percentage of Total Errors (%)", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Error Type", bbox_to_anchor=(1.05, 1), loc='upper left')

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 5:  # Only label if the chunk is big enough
            ax.text(x + width / 2, y + height / 2, f'{height:.1f}%',
                    horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()

    save_dir = get_plots_dir(conf, info)
    save_path = os.path.join(save_dir, "sid_stacked_bar.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_wer_vs_duration(info: Dict):
    conf = Parser()
    conf.get_args()
    local_info = info.copy()
    dataset_name = info['dataset']

    results_folder = get_results_path(conf, local_info)
    predictions_path = os.path.join(results_folder, 'predictions.csv')

    if not os.path.exists(predictions_path):
        print(f"No predictions found for {dataset_name} at {predictions_path}")
        return

    df = pd.read_csv(predictions_path)

    if 'duration' not in df.columns:
        print(f"Warning: 'duration' column not found in predictions for {dataset_name}. Update metrics.py.")
        return

    # Filter out extreme outliers (WER > 1.5) to keep the trendline accurate
    plot_df = df[df['wer'] <= 1.5]

    plt.figure(figsize=(10, 6))

    # Use seaborn's regplot to automatically calculate and plot the trendline
    sns.regplot(data=plot_df, x='duration', y='wer',
                scatter_kws={'alpha': 0.5, 'color': '#1f77b4'},
                line_kws={'color': 'red', 'linewidth': 2})

    # Add a vertical line at 30 seconds (Whisper's typical maximum context)
    plt.axvline(x=30.0, color='black', linestyle='--', alpha=0.7, label='Whisper 30s Limit')

    plt.title(f"WER vs. Audio Duration - {dataset_name}", fontsize=14)
    plt.xlabel("Audio Duration (Seconds)", fontsize=12)
    plt.ylabel("Word Error Rate (WER)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    save_dir = get_plots_dir(conf, info)
    save_path = os.path.join(save_dir, "wer_vs_duration.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_length_correlation(info: Dict):
    conf = Parser()
    conf.get_args()
    local_info = info.copy()
    dataset_name = info['dataset']

    results_folder = get_results_path(conf, local_info)
    predictions_path = os.path.join(results_folder, 'predictions.csv')

    if not os.path.exists(predictions_path):
        print(f"No predictions found for {dataset_name} at {predictions_path}")
        return

    df = pd.read_csv(predictions_path)

    # FIX: Handle pandas NaN loading for empty strings
    df['reference'] = df['reference'].fillna('')
    df['prediction'] = df['prediction'].fillna('')

    df['ref_len'] = df['reference'].str.len()
    df['pred_len'] = df['prediction'].str.len()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='ref_len', y='pred_len', alpha=0.5)

    # FIX: Safe max calculation in case df is entirely empty
    max_val = max(df['ref_len'].max(), df['pred_len'].max()) if not df.empty else 100

    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal (y=x)')
    plt.xlabel("Reference Length (chars)")
    plt.ylabel("Prediction Length (chars)")
    plt.title(f"Length Correlation - {dataset_name}")
    plt.legend()
    plt.tight_layout()

    save_dir = get_plots_dir(conf, info)
    save_path = os.path.join(save_dir, "length_correlation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_wer_distribution(info: Dict):
    conf = Parser()
    conf.get_args()
    local_info = info.copy()
    dataset_name = info['dataset']

    results_folder = get_results_path(conf, local_info)
    predictions_path = os.path.join(results_folder, 'predictions.csv')

    if not os.path.exists(predictions_path):
        print(f"No predictions found for {dataset_name} at {predictions_path}")
        return

    df = pd.read_csv(predictions_path)

    # Prevent crash if WER column is somehow missing or empty
    if 'wer' not in df.columns or df.empty:
        return

    plot_df = df.copy()
    plot_df.loc[plot_df['wer'] > 1.0, 'wer'] = 1.1

    plt.figure(figsize=(10, 6))
    sns.histplot(plot_df['wer'], bins=22, kde=True, color='skyblue')
    plt.axvline(df['wer'].mean(), color='red', linestyle='--', label=f'Mean: {df["wer"].mean():.2f}')
    plt.xlabel("Word Error Rate (WER)")
    plt.title(f"WER Distribution - {dataset_name}")
    plt.legend()
    plt.tight_layout()

    save_dir = get_plots_dir(conf, info)
    save_path = os.path.join(save_dir, "wer_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def examine_worst_predictions(info: Dict, top_n=10, sort_metric='n_wer'):
    conf = Parser()
    conf.get_args()
    local_info = info.copy()
    dataset_name = info['dataset']

    results_folder = get_results_path(conf, local_info)
    predictions_path = os.path.join(results_folder, 'predictions.csv')

    if not os.path.exists(predictions_path):
        print(f"No predictions found for {dataset_name} at {predictions_path}")
        return

    df = pd.read_csv(predictions_path)

    # Prevent crash if WER column is somehow missing or empty
    if sort_metric not in df.columns or df.empty:
        return

    worst_errors = df.sort_values(by=sort_metric, ascending=False).head(top_n)

    print(f"🚀 Top {top_n} Worst Predictions (Sorted by '{sort_metric}' descending)\n" + "=" * 70)

    for _, row in worst_errors.iterrows():
        print(f"🔹 Index: {row['index']} | Duration: {row['duration']}s")
        print(f"   Reference:  {row['reference']}")
        print(f"   Prediction: {row['prediction']}")
        if 'wer' in sort_metric:
            print(f"   Metrics:    n_WER: {row['n_wer']:.2f} | WER: {row['wer']:.2f}")
        elif 'cer' in sort_metric:
            print(f"   Metrics:    n_CER: {row['n_cer']:.2f} | CER: {row['cer']:.2f}")

        print(
            f"   Breakdown:  Substitutions: {row['substitutions']} | Insertions: {row['insertions']} | Deletions: {row['deletions']}")
        print("-" * 70)


def plot_word_level_cross_attention(conf, info, cross_attentions, generated_ids, tokenizer, layer_idx=-1, sample_idx=0):
    # 1. Max-pooling across attention heads to isolate dominant phonetic heads
    layer_attn = cross_attentions[layer_idx, sample_idx].max(dim=0).values.numpy()
    sample_ids = generated_ids[sample_idx].cpu().numpy()

    # Filter out PAD tokens
    pad_id = tokenizer.pad_token_id
    valid_idx = [i for i, tok in enumerate(sample_ids) if tok != pad_id]

    sample_ids = sample_ids[valid_idx]
    layer_attn = layer_attn[valid_idx]

    words = []
    word_attentions = []

    current_word_ids = []
    current_attn = np.zeros(layer_attn.shape[1])

    # 2. Group by subwords and decode safely
    for token_id, attn in zip(sample_ids, layer_attn):
        token_str = tokenizer.decode([token_id])
        is_boundary = token_str.startswith(' ') or token_str in ['.', ',', '!', '?', ':', ';', '\n']

        if is_boundary and current_word_ids:
            word_str = tokenizer.decode(current_word_ids).strip()
            if word_str:
                words.append(word_str)
                word_attentions.append(current_attn / (np.max(current_attn) + 1e-9))

            current_word_ids = []
            current_attn = np.zeros(layer_attn.shape[1])

        current_word_ids.append(token_id)
        current_attn += attn

    if current_word_ids:
        word_str = tokenizer.decode(current_word_ids).strip()
        if word_str:
            words.append(word_str)
            word_attentions.append(current_attn / (np.max(current_attn) + 1e-9))

    heatmap_data = np.vstack(word_attentions)

    # 3. Crop Masked Time Steps (Padding removal)
    col_sums = heatmap_data.sum(axis=0)
    cum_sums = np.cumsum(col_sums)
    total_mass = cum_sums[-1]

    if total_mass > 0:
        cutoff_idx = np.searchsorted(cum_sums, 0.995 * total_mass)
        crop_idx = min(cutoff_idx + 3, heatmap_data.shape[1])
        heatmap_data = heatmap_data[:, :crop_idx]

    # Smooths discrete acoustic frame transitions (sigma=1.2 along frame axis)
    heatmap_data = ndimage.gaussian_filter1d(heatmap_data, sigma=1.2, axis=1)
    # Re-normalize row-wise for clear intensity
    row_maxes = heatmap_data.max(axis=1, keepdims=True)
    heatmap_data = np.where(row_maxes > 0, heatmap_data / row_maxes, 0)

    # 5. Plot Continuous Heatmap using plt.imshow with Bilinear Interpolation
    plt.figure(figsize=(12, 8))

    im = plt.imshow(
        heatmap_data,
        aspect='auto',
        cmap='viridis',
        interpolation='bilinear',  # Creates smooth, continuous gradient transitions
        origin='upper'
    )

    # Formatting Y-ticks to line up with continuous word rows
    plt.yticks(range(len(words)), words, fontsize=12, rotation=0)
    plt.xticks([])  # Hide frame indices for clean aesthetic

    plt.xlabel("Audio Time Stream ➔", fontsize=12)

    plt.tight_layout()

    save_dir = get_plots_dir(conf, info)
    save_path = os.path.join(save_dir, f"word_level_alignment_continuous_layer_{layer_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Continuous alignment plot saved successfully to: {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=['common_voice', 'fleurs', 'massive', 'voxpopuli', 'yodas'], help='datasets')
    parser.add_argument('--splits', nargs='+', type=str, default=['train', 'validation', 'test'], help='split sets')
    parser.add_argument('--machine', type=str, default='kronos')
    parser.add_argument('--datetime', type=str, default=None)
    parser.add_argument('--turn', type=str, default=None)
    parser.add_argument('--speech_encoder_id', type=str, default='openai/whisper-large-v3')
    parser.add_argument('--language_model_id', type=str, default='elte-nlp/Racka-4B')
    parser.add_argument('--model_type', type=str, default='dual_fusion_checkpoints')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--name', type=str, default='default', help='Name of the experiment')

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)
    DATASETS = args.datasets

    if args.model_name is None:
        model_name = (args.speech_encoder_id.split('/')[1] + '_' + args.language_model_id.split('/')[1])

    for split in args.splits:
        base_info = {
            'model_type': args.model_type,
            'model_name': model_name,
            'split': split,
            'machine': args.machine,
            'datetime': args.datetime,
            'turn': args.turn,
            'name': args.name
        }

        aggregate_results(base_info)

        for dataset in args.datasets:
            base_info['dataset'] = dataset
            examine_worst_predictions(base_info, top_n=100, sort_metric='cer')
            plot_wer_distribution(base_info)
            plot_length_correlation(base_info)
            plot_wer_vs_duration(base_info)

        plot_sid_stacked_bar(base_info)
