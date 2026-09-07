import json
import os
from typing import Dict, Optional
from datasets import Dataset, Audio, interleave_datasets, concatenate_datasets, DatasetDict

from preprocessing.split import splitter
import random
from preprocessing.normalize import normalize
from typing import List
from pathlib import Path
import csv
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

def make_data_module(dataset_names,
                     bad_folder: str,
                     filters: List[str],
                     micro_data: bool = False,
                     train_samples: Optional[int] = None,
                     eval_samples: Optional[int] = None,
                     do_interleave: bool = True,
                     temperature: float = 1.0,
                     randomize: bool = False,
                     seed: int = 42,
                     exp: int = 0,
                     normalize_type: str = 'default'):

    split = splitter(exp=exp)
    data = split.split(datasets=dataset_names)

    train_sets = get_data(data['train'],
                          bad_folder,
                          filters=filters,
                          split='train',
                          iters=train_samples if micro_data else None,
                          randomize=randomize,
                          has_duration=True,
                          seed=seed,
                          normalize_type=normalize_type)

    val_sets = get_data(data['validation'],
                        bad_folder,
                        filters=filters,
                        split='validation',
                        iters=eval_samples,
                        randomize=randomize,
                        has_duration=True,
                        seed=seed,
                        normalize_type=normalize_type)

    if do_interleave and len(train_sets) > 1:
        train = interleave(train_sets, temperature=temperature)
    else:
        train = concatenate(train_sets)

    if randomize:
        train = train.shuffle(seed=seed)

    val = concatenate(val_sets)

    dataset = DatasetDict(
        {
            "train": train,
            "eval": val
        }
    )

    return dataset

def get_typed_data(dataset,
                   audio_name: str,
                   text_name: str,
                   has_duration: bool = False,
                   randomize: bool = False,
                   seed: int = 42,
                   normalized: bool = True,
                   normalize_type: str = 'default'):

    if has_duration:
        hf_data = {
            audio_name: [x["audio_filepath"] for x in dataset],
            text_name: [x["text"] for x in dataset],
            "duration": [x["duration"] for x in dataset],
            "index": [x["index"] for x in dataset],
            "dataset_name": [x["dataset_name"] for x in dataset]
        }

    else:
        hf_data = {
            audio_name: [x["audio_filepath"] for x in dataset],
            text_name: [x["text"] for x in dataset],
            "index": [x["index"] for x in dataset],
            "dataset_name": [x["dataset_name"] for x in dataset]
        }

    hf_data = Dataset.from_dict(hf_data)

    if randomize:
        hf_data = hf_data.shuffle(seed=seed)

    if normalized:
        if normalize_type == 'default':
            normalize_f = normalize
        elif normalize_type == 'whisper':
            normalize_f = BasicTextNormalizer()

        hf_data = hf_data.map(
            lambda x: {text_name: [normalize_f(t) for t in x[text_name]]},
            batched=True,
            num_proc=4
        )

    hf_data = hf_data.cast_column(audio_name, Audio(sampling_rate=16000))

    return hf_data


def get_data(paths, bad_folder: str, process: bool = True,
             type: int = 1, iters: Optional[int] = None, has_duration: bool = False,
             randomize: bool = False, seed: int = 42, normalized: bool = True, norm_mono: bool = False,
             filters: List[str] = None, split: str = '', normalize_type: str = 'default'):

    datasets = {}
    for name, path in paths.items():
        manifest_path = os.path.join(path)

        if filters is not None:
            bad_filepaths = set()

            for filter in filters:
                blacklist = os.path.join(bad_folder, name, split, f"bad_by_{filter}.csv")
                if os.path.exists(blacklist):
                    with open(blacklist, 'r', encoding='utf-8') as cf:
                        reader = csv.DictReader(cf)
                        for row in reader:
                            if 'filepath' in row:
                                bad_filepaths.add(row['filepath'])

                else:
                    print(f'No Blacklist (!!!) file for filter: {filter}, dataset: {name}, split: {split}')
                    exit()

            with open(manifest_path, 'r', encoding='utf-8') as f:
                dataset = []
                for l, line in enumerate(f):
                    item = json.loads(line)
                    item['index'] = l
                    item['dataset_name'] = name
                    if item.get("audio_filepath") not in bad_filepaths:
                        dataset.append(item)

        else:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                dataset = []
                for l, line in enumerate(f):
                    item = json.loads(line)
                    item['index'] = l
                    item['dataset_name'] = name
                    dataset.append(item)

        if iters:
            n = iters if iters < len(dataset) else len(dataset)
            dataset = random.sample(dataset, n)

        if process:
            if type == 1:
                audio_name = 'audio'
                text_name = 'reference'
            elif type == 2:
                audio_name = 'speech'
                text_name = 'transcription'

            hf_data = get_typed_data(dataset,
                                     audio_name,
                                     text_name,
                                     has_duration,
                                     randomize,
                                     seed,
                                     normalized,
                                     normalize_type)

        else:
            hf_data = dataset

        datasets[name] = hf_data

    return datasets

def interleave(datasets: Dict, temperature: float = 1.0):
    sizes = {name: len(ds) for name, ds in datasets.items()}

    weights = {d: s ** (1 / temperature) for d, s in sizes.items()}
    total = sum(weights.values())
    probs = {d: w / total for d, w in weights.items()}

    keys = datasets.keys()
    datasets = list(datasets.values())
    probs = [probs[key] for key in keys]

    merged_dataset = interleave_datasets(
        datasets,
        probabilities=probs,
        seed=42,
        stopping_strategy="all_exhausted"
    )

    return merged_dataset

def concatenate(datasets: Dict):
    merged_dataset = concatenate_datasets(
        list(datasets.values()),
    )

    return merged_dataset
