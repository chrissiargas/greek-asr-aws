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
import csv
from preprocessing.normalize import normalize


class StreamingSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, streaming_save_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_save_path = streaming_save_path
        self.is_main_process = int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

        if self.is_main_process and self.streaming_save_path:
            os.makedirs(os.path.dirname(self.streaming_save_path), exist_ok=True)
            with open(self.streaming_save_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["index", "reference", "prediction", "duration"])

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )

        if prediction_loss_only or generated_tokens is None:
            return loss, generated_tokens, labels

        if self.is_main_process and self.streaming_save_path:
            gen_ids = generated_tokens.detach().cpu().numpy()
            lbl_ids = labels.detach().cpu().numpy()

            gen_ids = np.where(gen_ids != -100, gen_ids, self.tokenizer.pad_token_id)
            preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            preds = [normalize(pd, with_signs=True) for pd in preds]

            lbl_ids = np.where(lbl_ids != -100, lbl_ids, self.tokenizer.pad_token_id)
            refs = self.tokenizer.batch_decode(lbl_ids, skip_special_tokens=True)
            refs = [normalize(rf, with_signs=True) for rf in refs]

            indices = inputs.get("index").cpu().numpy() if "index" in inputs else [None] * len(preds)
            durations = inputs.get("duration").cpu().numpy() if "duration" in inputs else [None] * len(preds)

            with open(self.streaming_save_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for idx, ref, pred, dur in zip(indices, refs, preds, durations):
                    writer.writerow([idx, ref, pred, dur])

        return loss, generated_tokens, labels


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


def evaluate_model(data, conf, args, info, checkpoint_path, checkpoint_dir, device='cuda'):
    training_args = conf.eval_args.training_args
    training_args['report_to'] = "none"
    if info['gen_kwargs'] is None:
        training_args['generation_config'] = GenerationConfig(**training_args['generation_config'])
    else:
        training_args['generation_config'] = info['gen_kwargs']

    training_args['disable_tqdm'] = True

    gen_config_obj = training_args.pop('generation_config', {})
    training_args = Seq2SeqTrainingArguments(**training_args)

    model = load_model(args, info, checkpoint_path, checkpoint_dir, device)
    tokenizer = model.language_tokenizer

    gen_kwargs = gen_config_obj.to_dict() if hasattr(gen_config_obj, "to_dict") else gen_config_obj
    model.language_model.generation_config.update(**gen_kwargs)
    model.generation_config = model.language_model.generation_config

    print('Generation Configurations:')
    print(model.generation_config)

    collator = DataCollator(processor=model.processor,
                            language_tokenizer=tokenizer,
                            padding='max_length',
                            truncation=True,
                            has_audio_lb_tokens=True,
                            has_duration_lb=args['predict_duration'],
                            to_chars=args['ctc'] or args['injection_downsample'] == 'cif',
                            contain_index=True)

    samples_path = os.path.join(info['res_folder'], "predictions.csv")

    trainer = StreamingSeq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_args,
        callbacks=[PredictionProgressCallback()],
        streaming_save_path=samples_path
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

    res_samples, res_total = get_metrics(predictions, references, indices, durations, verbose=True, is_whisper=True)

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


def get_results_path(conf, info, dataset, split='test', data_folder: bool = True):
    results_path = os.path.join(os.path.expanduser('~'),
                                conf.results_path,
                                'dual_fusion_checkpoints',
                                info['machine'],
                                info['model_name'],
                                info['datetime'],
                                str(info['turn']))

    if data_folder:
        if info['name'] is not None:
            results_path = os.path.join(results_path, str(dataset), split, info['name'])
        else:
            results_path = os.path.join(results_path, str(dataset), split)

    os.makedirs(results_path, exist_ok=True)

    return results_path


def get_bad_folder_path(conf):
    bad_folder_path = os.path.join(os.path.expanduser('~'),
                                   conf.dataset_path,
                                   conf.language,
                                   'bad_folder')

    return bad_folder_path


def evaluate(info, dataset, split='test', device='cuda', iters=None):
    conf, args, _, _, checkpoint_path, checkpoint_dir, _ = init(info, restart=False)

    info['res_folder'] = get_results_path(conf, info, dataset, split)
    bad_folder = get_bad_folder_path(conf)

    if not isinstance(dataset, list):
        dataset_names = [dataset]
    else:
        dataset_names = dataset

    split_manager = splitter(conf=conf)
    data = split_manager.split(datasets=dataset_names)

    evaluation_data = get_data(data[split],
                               bad_folder,
                               filters=FILTERS,
                               split=split,
                               iters=iters,
                               randomize=args.randomize,
                               has_duration=True,
                               normalize_type=args.normalize)

    evaluation_data = concatenate(evaluation_data)

    print(f"Evaluating on {dataset_names} ({len(evaluation_data)} samples)...")
    evaluate_model(evaluation_data, conf, args, info, checkpoint_path, checkpoint_dir, device)


FILTERS = []
from distutils.util import strtobool

if __name__ == "__main__":
    all_datasets = ['common_voice',
                    'fleurs',
                    'speech_massive',
                    'voxpopuli',
                    'yodas',
                    'dataocean_asr_657',
                    'dataocean_asr_659',
                    'datatang_asr_1',
                    'datatang_asr_2']

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str, default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets')
    parser.add_argument('--train_datasets', nargs='+', type=str, default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets of the trained models')
    parser.add_argument('--checkpoint_folder', type=str, default='dual_fusion_checkpoints')
    parser.add_argument('--speech_encoder_id', type=str, default='openai/whisper-large-v3')
    parser.add_argument('--language_model_id', type=str, default='ilsp/Llama-Krikri-8B-Instruct')
    parser.add_argument('--machine', type=str, default='kronos')
    parser.add_argument('--datetime', type=str, default=None)
    parser.add_argument('--turn', type=str, default=None)
    parser.add_argument('--exp', type=int, default=0, help='Path to config file')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--gen_kwargs', type=json.loads, default={})
    parser.add_argument('--concat_test', type=lambda x: bool(strtobool(x)), default=False)

    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    DATASETS = args.datasets
    model_name = (args.speech_encoder_id.split('/')[1] + '_' + args.language_model_id.split('/')[1])

    args_dict = {
        'res_folder': None,
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
        'name': args.name,
        'gen_kwargs': args.gen_kwargs,
        'concat_test': args.concat_test
    }

    local_rank, device = init_gpu()
    resume_wandb(local_rank, args_dict['datetime'], args_dict['model_name'], args_dict['exp'])

    try:
        if args_dict['concat_test']:
            args_dict['test_dataset'] = DATASETS
            evaluate(args_dict, DATASETS, device=device)
        else:
            for dataset in DATASETS:
                args_dict['test_dataset'] = dataset
                evaluate(args_dict, dataset, device=device)

        wandb.finish()

    finally:
        if local_rank in [-1, 0] and wandb.run is not None:
            wandb.finish()