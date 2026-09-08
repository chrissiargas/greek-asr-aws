import os
from platform import machine

import safetensors.torch
import torch
from os.path import dirname, abspath
import sys

from plotly.graph_objs.layout.slider import currentvalue

from evaluations.utils import extract_word_level_attention, calculate_attention_metrics

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from speechLM_utils.environment import set_environment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
set_environment()

import transformers
transformers.logging.set_verbosity_error()

from preprocessing.split import splitter
from preprocessing.prepare import get_data
from preprocessing.prepare import concatenate
from speechLM_utils.data_collator import DataCollator
from transformers import Seq2SeqTrainingArguments, GenerationConfig, Seq2SeqTrainer
import numpy as np
from metrics import get_metrics
from speechLM_utils.model import get_model
import wandb
import json
from speechLM_utils.utils import init
from speechLM_utils.model import get_max_step
from utils import extract_word_level_attention, calculate_attention_metrics, get_plots_dir
import random
import pandas as pd
from tqdm import tqdm
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def load_model(args, info, checkpoint_path, checkpoint_dir=None, device='cuda'):
    print(f"Initializing model...")

    model, tokenizer = get_model(args, info, device)

    if checkpoint_dir is None:
        max_step = get_max_step(checkpoint_path)
        weights_checkpoint = os.path.join(checkpoint_path, f"checkpoint-{max_step}")
        print('Found checkpoint to load: ', weights_checkpoint)
    else:
        weights_checkpoint = checkpoint_dir

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

        clean_state_dict = {
            k: v for k, v in state_dict.items()
            if 'bitsandbytes' not in k and 'quant_map' not in k and 'absmax' not in k
        }

        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
        print(f"[{device}] Weights loaded from {weights_checkpoint}.")
        print(f"[{device}] Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")
    except Exception as e:
        print(f"[{device}] Error: Could not load weights. {e}")

    model.eval()
    return model

def get_bad_folder_path(conf):
    bad_folder_path = os.path.join(os.path.expanduser('~'),
                                   conf.dataset_path,
                                   conf.language,
                                   'bad_folder')

    return bad_folder_path

def plot_word_level_attention(conf, info, heatmap_data, words, layer_idx=0):
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


def visualize_random_instance(info, dataset, split='test', num_samples: int = 200):
    conf, args, _, _, checkpoint_path, checkpoint_dir, _ = init(info, restart=False)

    print(f"Loading the Model...")

    training_args = conf.eval_args.training_args
    training_args['report_to'] = "none"
    if info['gen_kwargs'] is None:
        training_args['generation_config'] = GenerationConfig(**training_args['generation_config'])
    else:
        training_args['generation_config'] = info['gen_kwargs']

    training_args['disable_tqdm'] = True

    gen_config_obj = training_args.pop('generation_config', {})
    training_args = Seq2SeqTrainingArguments(**training_args)

    model = load_model(args, info, checkpoint_path, checkpoint_dir)
    tokenizer = model.language_tokenizer

    gen_kwargs = gen_config_obj.to_dict() if hasattr(gen_config_obj, "to_dict") else gen_config_obj
    model.language_model.generation_config.update(**gen_kwargs)
    model.generation_config = model.language_model.generation_config

    collator = DataCollator(processor=model.processor,
                            language_tokenizer=tokenizer,
                            padding='max_length',
                            truncation=True,
                            has_audio_lb_tokens=True,
                            has_duration_lb=args['predict_duration'],
                            to_chars=args['ctc'] or args['injection_downsample'] == 'cif',
                            contain_index=True)

    print(f"Model Loaded. Preparing the instance for inference...")

    if not isinstance(dataset, list):
        dataset_names = [dataset]
    else:
        dataset_names = dataset

    bad_folder = get_bad_folder_path(conf)

    split_manager = splitter(conf=conf)
    data = split_manager.split(datasets=dataset_names)

    evaluation_data = get_data(data[split],
                              bad_folder,
                              filters=FILTERS,
                              split=split,
                              iters=None,
                              randomize=True,
                              has_duration=True,
                              normalize_type=args.normalize)

    evaluation_data = concatenate(evaluation_data)
    num_samples = min(num_samples, len(evaluation_data))
    print(f"\n--- Evaluating Attention Metrics on {num_samples} samples from {dataset} ---")

    metrics_records = []

    for q in tqdm(range(num_samples), desc="Computing Attention Dynamics"):
        random_idx = random.randint(0, len(evaluation_data) - 1)
        instance = evaluation_data[random_idx]

        batch = collator([instance])

        audios = batch['audios']
        audio_masks = batch['audio_masks']

        # We pass return_attention=True to trigger the tracking mechanism we added to CrossAttention
        outputs, cross_attentions = model.generate(
            audios=audios,
            audio_masks=audio_masks,
            max_new_tokens=200,
            return_attention=True
        )

        generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs

        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Prediction: {prediction}\n")

        for layer_idx in [0,1,2]:
            print(f"Plotting Cross-Modal Alignment for Layer {layer_idx}...")
            heatmap_data, _ = extract_word_level_attention(
                cross_attentions=cross_attentions,
                generated_ids=generated_ids,
                tokenizer=tokenizer,
                layer_idx=layer_idx,  # You can change this to 0 or 1 depending on how many injection layers you have
                sample_idx=0
            )

            res = calculate_attention_metrics(heatmap_data)
            metrics_records.append({
                'sample_idx': q,
                'layer_idx': layer_idx,
                'mean_entropy': res['mean_entropy'],
                'norm_entropy': res['norm_entropy'],
                'diagonality_r': res['diagonality_r']
            })

        df_metrics = pd.DataFrame(metrics_records)

        summary = df_metrics.groupby('layer_idx').agg({
            'norm_entropy': ['mean', 'std'],
            'diagonality_r': ['mean', 'std']
        }).round(3)

        print("\n================ Layer Attention Dynamics Summary ================")
        print(summary)
        print("==================================================================")

        # Save to disk
        plots_dir = os.path.join(os.path.expanduser('~'), conf.results_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        summary_path = os.path.join(plots_dir, f"{dataset}_attention_dynamics_summary.csv")
        summary.to_csv(summary_path)
        print(f"Metrics table saved to: {summary_path}")

        return summary



FILTERS = ['duration', 'length']
from distutils.util import strtobool
import argparse
import os

all_datasets = ['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, default=all_datasets, help='datasets')
    parser.add_argument('--train_datasets', nargs='+', type=str, default=all_datasets, help='datasets of the trained models')
    parser.add_argument('--checkpoint_folder', type=str, default='dual_fusion_checkpoints')
    parser.add_argument('--speech_encoder_id', type=str, default='openai/whisper-large-v3')
    parser.add_argument('--language_model_id', type=str, default='ilsp/Llama-Krikri-8B-Instruct')
    parser.add_argument('--machine', type=str, default='kronos')
    parser.add_argument('--datetime', type=str, default=None)
    parser.add_argument('--turn', type=str, default=None)
    parser.add_argument('--exp', type=int, default=0, help='Path to config file')
    parser.add_argument('--gen_kwargs', type=json.loads, default={})

    args, unknown = parser.parse_known_args()

    DATASETS = args.datasets
    model_name = (args.speech_encoder_id.split('/')[1] + '_' + args.language_model_id.split('/')[1])

    args_dict = {
        'train_dataset': args.train_datasets,
        'checkpoint_folder': args.checkpoint_folder,
        'model_name': model_name,
        'speech_encoder_id': args.speech_encoder_id,
        'language_model_id': args.language_model_id,
        'bit4': True,
        'machine': args.machine,
        'datetime': args.datetime,
        'turn': args.turn,
        'exp': args.exp,
        'gen_kwargs': args.gen_kwargs
    }

    for dataset in DATASETS:
        args_dict['test_dataset'] = dataset
        visualize_random_instance(args_dict, dataset)