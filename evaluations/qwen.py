import os
from os.path import dirname, abspath
import sys
import logging

logging.disable(logging.CRITICAL)
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from speechLM_utils.environment import set_environment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
set_environment()

import torch
import librosa
from tqdm import tqdm
import argparse
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from preprocessing.split import splitter
from preprocessing.prepare import get_data, concatenate
from speechLM_utils.utils import init_gpu, resume_wandb, init
from preprocessing.normalize import normalize
import jiwer
import pandas as pd

def get_prediction(x, device, model, processor):
    path = x['audio_filepath']
    index = x['index']
    duration = x['duration']

    try:
        audio, sr = librosa.load(path, sr=16000)
        messages = [
            {"role": "system", "content": "You are an ASR assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": path},
                {"type": "text", "text": "Μετάγραψε το ηχητικό απόσπασμα στα Ελληνικά με απόλυτη ακρίβεια."}
            ]}
        ]

        reference = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=reference, audios=audio, return_tensors="pt", padding=True).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        n_prediction = normalize(prediction, with_signs=False)
        n_reference = normalize(reference, with_signs=False)

        prediction = normalize(prediction, with_signs=True)
        reference = normalize(reference, with_signs=True)

        wer = jiwer.wer(reference, prediction)
        n_wer = jiwer.wer(n_reference, n_prediction)
        cer = jiwer.cer(reference, prediction)
        n_cer = jiwer.cer(n_reference, n_prediction)
        ratio = len(prediction) / len(reference) if len(reference) > 0 else 0
        n_ratio = len(n_prediction) / len(n_reference) if len(reference) > 0 else 0

        out = jiwer.process_words(reference, prediction)
        sub_ = out.substitutions
        in_ = out.insertions
        del_ = out.deletions

        return (reference, prediction, n_reference, n_prediction,
                (index, wer, n_wer, cer, n_cer, ratio, n_ratio, sub_, in_, del_, duration))

    except Exception as e:
        print(f"Error processing {path}: {e}")

def get_eval_metrics(data, res_folder, device = 'cuda'):
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=None  # Load directly to specific GPU
    ).to(device)
    model.eval()

    references, predictions = [], []
    n_references, n_predictions = [], []
    wers, cers, ratios = [], [], []
    n_wers, n_cers, n_ratios = [], [], []
    subs, ins, dels = [], [], []
    indices, durations = [], []

    progress_bar = tqdm(
        data, leave=True, ncols=100, colour='blue'
    )

    for entry in progress_bar:
        reference, prediction, n_reference, n_prediction, metrics  = get_prediction(entry, device, model, processor)
        index, wer, n_wer, cer, n_cer, ratio, n_ratio, sub_, in_, del_, duration = metrics

        references.append(reference)
        predictions.append(prediction)
        n_references.append(n_reference)
        n_predictions.append(n_prediction)
        wers.append(wer)
        n_wers.append(n_wer)
        cers.append(cer)
        n_cers.append(n_cer)
        ratios.append(ratio)
        n_ratios.append(n_ratio)
        subs.append(sub_)
        ins.append(in_)
        dels.append(del_)
        indices.append(index)
        durations.append(duration)

    results_per_instance = {'index': indices,
                            'reference': references,
                            'prediction': predictions,
                            'wer': wers,
                            'n_wer': n_wers,
                            'cer': cers,
                            'n_cer': n_cers,
                            'ratio': ratios,
                            'n_ratio': n_ratios,
                            'substitutions': subs,
                            'insertions': ins,
                            'deletions': dels,
                            'duration': durations}

    results_per_instance = pd.DataFrame(results_per_instance)

    predictions_path = os.path.join(res_folder, f"predictions.csv")
    results_per_instance.to_csv(predictions_path)

    total_wer = jiwer.wer(references, predictions)
    total_n_wer = jiwer.wer(n_references, n_predictions)
    total_cer = jiwer.cer(references, predictions)
    total_n_cer = jiwer.cer(n_references, n_predictions)
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

    results_path = os.path.join(res_folder, f"results.csv")
    total_results.to_csv(results_path)

    return predictions_path, results_path

def get_results_path(conf, data_name, data_folder: bool = True):
    results_path = os.path.join(os.path.expanduser('~'),
                                conf.results_path,
                                'qwen2audio')

    if data_folder:
        results_path = os.path.join(results_path, data_name)

    os.makedirs(results_path, exist_ok=True)

    return results_path

def get_bad_folder(conf, data_name):
    bad_folder = os.path.join(os.path.expanduser('~'),
                              conf.dataset_path,
                              conf.language,
                              'bad_folder')

    return bad_folder

def evaluate(conf, dataset, set='test', device='cuda', iters = None):
    bad_folder = get_bad_folder(conf, dataset)
    res_folder = get_results_path(conf, dataset)

    if type(dataset) is not list:
        dataset_names = [dataset]
    else:
        dataset_names = dataset

    split = splitter()
    data = split.split(datasets=dataset_names)
    data_set = get_data(data[set], bad_folder, iters=iters, normalized=False, has_duration=True, filters=FILTERS, split=set)
    data_set = concatenate(data_set)

    print(f"Evaluating on {dataset_names} ({len(data_set)} samples)...")
    get_eval_metrics(dataset, res_folder, device)

FILTERS = ['duration', 'length']
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str, default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets')

    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    DATASETS = args.datasets

    local_rank, device = init_gpu()

    for dataset in DATASETS:
        evaluate(dataset,
                 set='test',
                 device=device)


