import os
from platform import machine

import safetensors.torch
import torch
from os.path import dirname, abspath
import sys

from plotly.graph_objs.layout.slider import currentvalue

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from speechLM_utils.environment import set_environment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
set_environment()

import transformers
transformers.logging.set_verbosity_error()

from config.parser import Parser
from preprocessing.split import splitter
from preprocessing.prepare import get_data
from preprocessing.prepare import concatenate
from speechLM_utils.data_collator import DataCollator
from speechLM_utils.train import get_checkpoint
from transformers import Seq2SeqTrainingArguments, GenerationConfig, Seq2SeqTrainer
import numpy as np
from metrics import get_metrics
from speechLM_utils.model import get_model
import wandb
import json
from transformers import TrainerCallback
from tqdm.auto import tqdm
from speechLM_utils.utils import init_gpu, resume_wandb, init
from speechLM_utils.model import get_max_step
import argparse

class PredictionProgressCallback(TrainerCallback):
    def __init__(self):
        self.prediction_bar = None
        # Strictly rely on the OS environment variable you already trust
        self.is_main_process = int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

    def on_prediction_step(self, args, state, control, **kwargs):
        if self.is_main_process:
            if self.prediction_bar is None:
                eval_dataloader = kwargs.get("eval_dataloader", None)
                total_batches = len(eval_dataloader) if eval_dataloader else None

                # Force output to stdout so multi-GPU launchers don't swallow it
                self.prediction_bar = tqdm(
                    total=total_batches,
                    desc="Evaluating Batches",
                    leave=True,
                    file=sys.stdout
                )

            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if self.is_main_process and self.prediction_bar is not None:
            self.prediction_bar.close()
            self.prediction_bar = None

def load_model(model_type, args, info, checkpoint_path, checkpoint_dir=None, device='cuda'):
    print(f"Initializing model...")

    model, tokenizer = get_model(model_type, args, info, device)

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

def debug_predictions(predictions, references, generated_ids, label_ids, tokenizer):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    for pred, ref, gen, label in zip(predictions, references, generated_ids, label_ids):
        gen_clean = [token for token in gen if token != pad_id]
        label_clean = [token for token in label if token != -100 and token != pad_id]

        print(pred)
        print(ref)
        print()

        gen_len = len(gen_clean)
        ref_len = len(label_clean)

        print(gen_clean)
        print(label_clean)
        print()

        print(gen_len)
        print(ref_len)


def get_eval_metrics(data, model_type, conf, args, info, checkpoint_path, checkpoint_dir, device = 'cuda'):
    training_args = conf.eval_args.training_args
    training_args['report_to'] = "none"
    training_args['generation_config'] = GenerationConfig(**training_args['generation_config'])
    training_args['disable_tqdm'] = True
    training_args = Seq2SeqTrainingArguments(**training_args)

    gen_config_dict = training_args.pop('generation_config', {})

    model = load_model(model_type, args, info, checkpoint_path, checkpoint_dir, device)
    tokenizer = model.language_tokenizer

    model.generation_config.update(**gen_config_dict)

    collator = DataCollator(processor=model.processor,
                            language_tokenizer=tokenizer,
                            padding='max_length',
                            truncation=True,
                            has_audio_lb_tokens=True,
                            has_duration_lb=args['predict_duration'],
                            to_chars=args['ctc'] or args['injection_downsample'] == 'cif',
                            contain_index=False)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_args,
        callbacks=[PredictionProgressCallback()]
    )

    print(f"Starting inference on {len(data)} samples...")

    output = trainer.predict(data)
    indices = data['index']
    durations = data['duration']

    generated_ids = output.predictions
    label_ids = output.label_ids

    data_len = len(indices)
    generated_ids = generated_ids[:data_len]
    label_ids = label_ids[:data_len]

    generated_ids = np.where(generated_ids != -100, generated_ids, tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # debug_predictions(predictions, references, generated_ids, label_ids, tokenizer)

    print(f"Computing metrics...")

    res_samples, res_total = get_metrics(predictions, references, indices, durations, with_total_metrics=True)

    samples_path = os.path.join(info['res_folder'], "predictions.csv")
    total_path = os.path.join(info['res_folder'], "results.csv")

    res_samples.to_csv(samples_path)
    res_total.to_csv(total_path)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [-1, 0] and wandb.run is not None:
        dataset_name = info['test_dataset']
        current_step = int(info.get('turn', 0))

        wandb.log({
            f"{dataset_name}/predictions": wandb.Table(dataframe=res_samples),
            f"{dataset_name}/results": wandb.Table(dataframe=res_total)
        }, step=current_step)

    return samples_path, total_path

def get_results_path(conf, info, data_folder: bool = True):
    results_path = os.path.join(os.path.expanduser('~'),
                                conf.results_path,
                                info['model_type'],
                                info['machine'],
                                info['model_name'],
                                info['datetime'],
                                str(info['turn']))

    if data_folder:
        results_path = os.path.join(results_path, info['test_dataset'])

    os.makedirs(results_path, exist_ok=True)

    return results_path

def evaluate(info, set='test', device='cuda', iters = None):
    conf, args, _, bad_folder, _, checkpoint_path, checkpoint_dir, _ = init(info['model_type'], info, restart=False)

    res_folder = get_results_path(conf, info)
    info['res_folder'] = res_folder

    if type(info['test_dataset']) is not list:
        dataset_names = [info['test_dataset']]
    else:
        dataset_names = info['test_dataset']

    split = splitter()
    data = split.split(datasets=dataset_names)
    data_set = get_data(data[set], bad_folder, iters=iters, normalized=False, has_duration=True, filters=FILTERS, split=set)
    data_set = concatenate(data_set)

    print(f"Evaluating on {dataset_names} ({len(data_set)} samples)...")
    get_eval_metrics(data_set, info['model_type'], conf, args, info, checkpoint_path, checkpoint_dir, device)

FILTERS = ['duration', 'length']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str, default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets')
    parser.add_argument('--model_type', type=str, default='dual_fusion')
    parser.add_argument('--train_datasets', nargs='+', type=str, default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets of the trained models')
    parser.add_argument('--checkpoint_folder', type=str, default='dual_fusion_checkpoints')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--speech_encoder_id', type=str, default='openai/whisper-large-v3')
    parser.add_argument('--language_model_id', type=str, default='ilsp/Llama-Krikri-8B-Instruct')
    parser.add_argument('--machine', type=str, default='kronos')
    parser.add_argument('--datetime', type=str, default=None)
    parser.add_argument('--turn', type=str, default=None)
    parser.add_argument('--stage', type=int, default=None)

    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    DATASETS = args.datasets

    base_info = {
        'model_type': args.model_type,
        'res_folder': None,
        'train_dataset': args.train_datasets,
        's_': False,
        'checkpoint_folder': args.checkpoint_folder,
        'model_name': args.model_name,
        'speech_encoder_id': args.speech_encoder_id,
        'language_model_id': args.language_model_id,
        'bit4': True,
        'machine': args.machine,
        'datetime': args.datetime,
        'turn': args.turn,
        'stage': args.stage
    }

    if base_info['datetime'] is None:
        checkpoints_path = os.path.join(os.path.expanduser('~'),
                                        'cache',
                                        'checkpoints',
                                        base_info['model_type'],
                                        base_info['machine'],
                                        base_info['model_name'])

        for datetime_folder in os.listdir(checkpoints_path):
            base_info['datetime'] = datetime_folder.split('@')[-1]
            turns = os.listdir(os.path.join(checkpoints_path, datetime_folder))
            base_info['turn'] = turns[0].replace('checkpoint-', '')

            print('\n')
            print('---------------------------------------------------------')
            print(base_info['machine'])
            print(base_info['datetime'])
            print(base_info['turn'])
            print('---------------------------------------------------------')

            local_rank, device = init_gpu()
            resume_wandb(local_rank, base_info)

            try:
                for dataset in DATASETS:
                    base_info['test_dataset'] = dataset

                    evaluate(base_info,
                             set='test',
                             device=device,
                             iters=None)

                wandb.finish()

            finally:
                if local_rank in [-1, 0] and wandb.run is not None:
                    wandb.finish()

    else:
        local_rank, device = init_gpu()
        resume_wandb(local_rank, base_info)

        try:
            for dataset in DATASETS:
                base_info['test_dataset'] = dataset

                evaluate(base_info,
                         set='test',
                         device=device,
                         iters=None)

            wandb.finish()

        finally:
            if local_rank in [-1, 0] and wandb.run is not None:
                wandb.finish()