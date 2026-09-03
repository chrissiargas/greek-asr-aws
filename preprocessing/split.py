from config.parser import Parser
import os
import json
from pathlib import Path
from typing import Dict, Optional, List
import random
import numpy as np

class splitter:
    def __init__(self, conf: Optional[Parser] = None, validation: bool = True, exp: int = 0):
        # if conf is None:
        self.conf = Parser()
        self.conf.get_args(exp)
        # else:
        #     self.conf = conf

        self.validation = validation
        self.seed = 42

    def perform_speaker_split(self, manifest_folder: str, manifest_file: str, dataset: str, test_split: float = 0):
        print(f"[{dataset}] No pre-existing splits found. Performing Speaker-Disjoint Split on the fly...")

        data = pd.read_json(manifest_file, lines=True)

        if 'subject' in data.columns and not data['subject'].isnull().all():
            group_col = 'subject'
        elif 'session' in data.columns:
            group_col = 'session'
        else:
            raise ValueError(f"[{dataset}] Error: Neither 'subject' nor 'session' found in the manifest!")

        print(f"[{dataset}] Grouping data by: '{group_col}'")

        group_stats = data.groupby(group_col, as_index=False)['duration'].sum()

        group_stats['cum_duration'] = group_stats['duration'].cumsum()
        total_duration = group_stats['duration'].sum()

        train_threshold = (1. - 2. * test_split) * total_duration
        val_threshold = train_threshold + (test_split * total_duration)

        conditions = [
            group_stats['cum_duration'] <= train_threshold,
            (group_stats['cum_duration'] > train_threshold) & (group_stats['cum_duration'] <= val_threshold)
        ]
        choices = ['train', 'validation']
        group_stats['split'] = np.select(conditions, choices, default='test')

        splits_dict = group_stats.groupby('split')[group_col].apply(list).to_dict()

        print(f"Train Subjects: {splits_dict.get('train', [])}\n"
              f"Validation Subjects: {splits_dict.get('validation', [])}\n"
              f"Test Subjects: {splits_dict.get('test', [])}\n")

        data = data.merge(group_stats[[group_col, 'split']], on=group_col, how='left')

        generated_paths = {}
        for split_name in ['train', 'validation', 'test']:
            split_data = data[data['split'] == split_name].drop(columns=['split'])
            print(f"{split_name} samples: {len(split_data)}")

            output_path = os.path.join(manifest_folder, f'{self.conf.language}_{split_name}_default.json')
            split_data.to_json(output_path, orient='records', lines=True, force_ascii=False)
            generated_paths[split_name] = output_path

        print(f"[{dataset}] Successfully generated splits.")
        return generated_paths

    def perform_random_split(self, manifest_folder: str, manifest_file: str, dataset: str, validation_split: float = 0.09, test_split: float = 0.01):
        print(f"[{dataset}] No pre-existing splits found. Performing Random Split on the fly...")

        data = pd.read_json(manifest_file, lines=True)

        total_samples = len(data)
        train_size = int((1 - validation_split - test_split) * total_samples)
        val_size = int(validation_split * total_samples)

        shuffled_data = data.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        train_data = shuffled_data.iloc[:train_size]
        val_data = shuffled_data.iloc[train_size:train_size + val_size]
        test_data = shuffled_data.iloc[train_size + val_size:]

        generated_paths = {}
        for split_name, split_data in zip(['train', 'validation', 'test'], [train_data, val_data, test_data]):
            print(f"{split_name} samples: {len(split_data)}")

            output_path = os.path.join(manifest_folder, f'{self.conf.language}_{split_name}_random.json')
            split_data.to_json(output_path, orient='records', lines=True, force_ascii=False)
            generated_paths[split_name] = output_path

        print(f"[{dataset}] Successfully generated splits.")
        return generated_paths

    def merge_manifests(self, manifest_files: Dict, split: str):
        output_path = os.path.join(self.conf.dataset_path, split +  '.json')

        if Path(output_path).exists():
            os.remove(output_path)

        split_files = manifest_files[split]
        for dataset, file in split_files.items():
            if not os.path.exists(file):
                print(f"Warning: File not found - {file}")
                continue

            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    with open(output_path, 'a', encoding='utf-8') as g:
                        entry = json.loads(line)
                        entry['dataset'] = dataset
                        entry['split'] = split
                        g.write(json.dumps(entry) + '\n')

    def split(self, validation: bool = True, merging: bool = False, datasets: Optional[List] = None):
        manifests = {
            'train': {},
            'validation': {},
            'test': {}
        }

        datasets = datasets if datasets else self.conf.datasets

        for dataset in datasets:
            manifest_folder = os.path.join(os.path.expanduser('~'), self.conf.dataset_path, self.conf.language, dataset, 'manifests')
            for split in ['train', 'validation', 'test']:
                manifest_file = os.path.join(manifest_folder, f'{self.conf.language}_{split}.json')
                if not os.path.exists(manifest_file):
                    print(f"Warning: File not found - {manifest_file}")
                    continue

                manifests[split][dataset] = manifest_file

        if not validation:
            manifests['train'].update(manifests['validation'])
            del manifests['validation']

        if merging:
            self.merge_manifests(manifests, 'train')
            if validation:
                self.merge_manifests(manifests, 'validation')
            self.merge_manifests(manifests, 'test')

        return manifests

    def split_for_train(self):
        if self.conf.splitting:
            self.split(validation=True)

    def split_for_inference(self):
        if self.conf.splitting:
            self.split(validation=True)

import pandas as pd
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    B = splitter(splitting=True, validation=True)
    B.split_for_train()

    df = pd.read_json(os.path.join(B.conf.dataset_path, 'train.json'), lines=True)
    common_voice = df[df['dataset'] == 'common_voice']
    print(common_voice)




