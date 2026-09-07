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

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, SpeechT5ForSpeechToText, SpeechT5Processor, \
    AutoModelForCTC
from preprocessing.prepare import get_data

from config.parser import Parser
from preprocessing.split import splitter
from speechLM_utils.utils import init_gpu, resume_wandb, init
from typing import Optional
import os
from preprocessing.prepare import concatenate
import numpy as np
from tqdm import tqdm
from preprocessing.parallel import parallelize_process, concat_dataframes
from evaluations.metrics import get_metrics
import json


def gpu_evaluate(data, gpu_id, args):
    data = Dataset.from_dict(data)

    torch_dtype = torch.float16
    batch_size = 8

    model = AutoModelForSpeechSeq2Seq.from_pretrained(args['model_name'], torch_dtype=torch_dtype).to(
        f"cuda:{gpu_id}")
    processor = AutoProcessor.from_pretrained(args['model_name'])

    base_gen_kwargs = {
        "language": "el",
        "task": "transcribe",
        "return_timestamps": False,
        "num_beams": 5,
        "condition_on_prev_tokens": False,
        "repetition_penalty": 1.15,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 4,
        "temperature": 0.0

    }

    if len(base_gen_kwargs) > 0:
        base_gen_kwargs.update(**args['gen_kwargs'])

    pipe_kwargs = {
        "model": model,
        "tokenizer": processor.tokenizer,
        "feature_extractor": processor.feature_extractor,
        "torch_dtype": torch_dtype,
        "device": f"cuda:{gpu_id}",
        "chunk_length_s": 30.0,
        "batch_size": batch_size,
    }

    if len(base_gen_kwargs) > 0:
        pipe_kwargs["generate_kwargs"] = base_gen_kwargs

    pipe = pipeline(
        "automatic-speech-recognition",
        **pipe_kwargs
    )

    predictions = []

    def yield_data():
        for i, item in enumerate(data):
            audio_data = item["audio"]

            if isinstance(audio_data["array"], list):
                audio_data["array"] = np.array(audio_data["array"], dtype=np.float32)
            yield audio_data

    iter_kwargs = {"batch_size": batch_size}
    if len(base_gen_kwargs) > 0:
        iter_kwargs["generate_kwargs"] = base_gen_kwargs

    for out in tqdm(pipe(yield_data(), **iter_kwargs),
                    position=gpu_id,
                    total=len(data),
                    desc=f"GPU {gpu_id}"):
        predictions.append(out['text'].strip())

    references = [t.strip() for t in data["reference"]]

    indices = data['index']
    durations = data['duration']
    res_samples, _ = get_metrics(predictions, references, indices, durations, verbose=False, is_whisper=True)

    samples_path = os.path.join(args['res_folder'], f"predictions_{gpu_id}.csv")
    res_samples.to_csv(samples_path)

    return samples_path


def get_results_path(conf, args, dataset: str, split: str):
    results_path = os.path.join(os.path.expanduser('~'),
                                conf.results_path,
                                args['model_type'],
                                args['model_name'],
                                dataset,
                                split,
                                args['name'])

    os.makedirs(results_path, exist_ok=True)

    return results_path


def get_bad_folder_path(conf, dataset: str, split: str):
    bad_folder_path = os.path.join(os.path.expanduser('~'),
                                   conf.dataset_path,
                                   conf.language,
                                   'bad_folder')

    return bad_folder_path


def get_total_metrics(base_filename: str):
    merged_df = pd.read_csv(f"{base_filename}.csv")

    predictions = merged_df['prediction'].fillna("").astype(str).tolist()
    references = merged_df['reference'].fillna("").astype(str).tolist()
    indices = merged_df['index'].tolist()
    durations = merged_df['duration'].tolist()

    res_samples, total_results = get_metrics(predictions, references, indices, durations)

    res_samples.to_csv(f"{base_filename}.csv", index=False)
    total_results.to_csv(f"{base_filename}_total_metrics.csv", index=False)


def evaluate(args, dataset: str, split: str = 'test'):
    conf = Parser()
    conf.get_args()

    args['res_folder'] = get_results_path(conf, args, dataset, split)
    bad_folder = get_bad_folder_path(conf, dataset, split)

    if not isinstance(dataset, list):
        dataset_names = [dataset]
    else:
        dataset_names = dataset

    split_manager = splitter()
    data = split_manager.split(datasets=dataset_names)
    evaluation_data = get_data(data[split],
                               bad_folder,
                               normalized=False,
                               has_duration=True,
                               filters=FILTERS,
                               split=split,
                               normalize_type='whisper')
    evaluation_data = concatenate(evaluation_data)

    print(f"Evaluating on {dataset} ({len(evaluation_data)} samples)...")

    torch.multiprocessing.set_start_method('spawn', force=True)
    parallelize_process(evaluation_data, gpu_evaluate, gpus=[0, 1, 2, 3], info=args)

    print(f"Merging GPU prediction files for {dataset}...")
    base_filename = os.path.join(args['res_folder'], "predictions")

    concat_dataframes(base_filename, remove=True)
    print(f"✅ Final merged predictions saved to: {base_filename}.csv")

    get_total_metrics(base_filename)
    print(f"✅ Unified metrics saved to: {base_filename}_total_metrics.csv\n")


import argparse

FILTERS = []

if __name__ == "__main__":
    print('Starting...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets')
    parser.add_argument('--splits', nargs='+', type=str, default=['train', 'validation', 'test'], help='split sets')
    parser.add_argument('--model_type', type=str, default='whisper')
    parser.add_argument('--model_name', type=str, default='openai/whisper-large-v3')
    parser.add_argument('--test_split', type=float, default=0, help='test split percentage')
    parser.add_argument('--name', type=str, default='default', help='name of the experiment')
    parser.add_argument('--gen_kwargs', type=json.loads, default={})

    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args_dict = vars(args)

    for dataset in args.datasets:
        for split in args.splits:
            # local_rank, device = init_gpu()
            evaluate(args_dict, dataset, split)

