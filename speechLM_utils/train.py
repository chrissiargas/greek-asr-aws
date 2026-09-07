import warnings
import wandb
from lightning.fabric.utilities.distributed import group

warnings.filterwarnings("ignore")
import os
import sys
from os.path import dirname

sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))
from speechLM_utils.environment import set_environment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
set_environment()

import gc
from preprocessing.prepare import make_data_module
from speechLM_utils.data_collator import DataCollator, compute_length
import torch
import evaluate
import numpy as np
from transformers import EarlyStoppingCallback
from speechLM_utils.model import get_model, load_weights
import copy
from transformers.integrations import TensorBoardCallback
from speechLM_utils.trainers import MultiLossTrainer
from speechLM_utils.checkpoint import *
from speechLM_utils.metrics import wrap_compute_metrics
from speechLM_utils.utils import init_gpu, init_info, init, init_wandb
from distutils.util import strtobool
import json

# Load metrics once
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")
N_samples_for_metrics = 200


class OffsetTensorBoardCallback(TensorBoardCallback):
    def __init__(self, step_offset=0, tb_writer=None):
        super().__init__(tb_writer=tb_writer)
        self.step_offset = step_offset

    def on_log(self, args, state, control, logs=None, **kwargs):
        original_step = state.global_step
        state.global_step = original_step + self.step_offset
        super().on_log(args, state, control, logs=logs, **kwargs)
        state.global_step = original_step


def two_stage_train(dataset, args, training_args, info, checkpoint_path, checkpoint_dir, writer, device, exp: int = 0):
    if not SKIP_STAGE1:
        stage1_args = copy.deepcopy(args)

        print("--- STARTING STAGE 1: ADAPTER ALIGNMENT ---")

        stage1_args.linguistic_lora = False
        stage1_args.static_projector = False
        stage1_args.static_injection_layers = False

        stage1_training_args = copy.deepcopy(training_args)
        stage1_training_args.num_train_epochs = args.first_stage_epochs
        stage1_training_args.output_dir = os.path.join(checkpoint_path, "stage1")
        shared_logging_dir = os.path.join(checkpoint_path, "logs")
        stage1_training_args.logging_dir = shared_logging_dir

        train_model(dataset, stage1_args, stage1_training_args, info, stage1_training_args.output_dir, checkpoint_dir,
                    writer, device, exp=exp)

        print("--- STAGE 1 COMPLETE ---")

        torch.cuda.empty_cache()

    stage2_args = copy.deepcopy(args)

    print("--- STARTING STAGE 2: LORA REFINEMENT ---")

    stage2_args.linguistic_lora = True  # Turn LoRA ON
    stage2_args.static_projector = True
    stage2_args.static_injection_layers = True

    stage2_training_args = copy.deepcopy(training_args)
    stage2_training_args.num_train_epochs = args.second_stage_epochs
    previous_checkpoint = os.path.join(checkpoint_path, "stage1")
    stage2_training_args.output_dir = os.path.join(checkpoint_path, "stage2")
    shared_logging_dir = os.path.join(checkpoint_path, "logs")
    stage2_training_args.logging_dir = shared_logging_dir

    train_model(dataset, stage2_args, stage2_training_args, info, previous_checkpoint,
                stage2_training_args.output_dir, writer, device, load=True, exp=exp)

    print("--- STAGE 2 COMPLETE ---")


def train_model(dataset, args, training_args, info, checkpoint_path, checkpoint_dir, writer, device, load: bool = False,
                exp: int = 0):
    model, tokenizer = get_model(args, info, device, exp, ATTN_IMPL)

    if load:
        model, max_step = load_weights(model, checkpoint_path, args, device)

    processor = model.processor
    tokenizer = model.language_tokenizer

    padding = 'max_length'
    truncation = True

    collator = DataCollator(processor=processor, language_tokenizer=tokenizer,
                            padding=padding, truncation=truncation,
                            has_audio_lb_tokens=True,
                            has_duration_lb=args['predict_duration'],
                            to_chars=args['ctc'] or args['injection_downsample'] == 'cif',
                            prompt_verbatim=args['prompt_verbatim'])

    params = model.get_named_params()

    optimizer_grouped_parameters = [group for group in [
        {"params": params['lora'], "lr": args.lora_lr},
        {"params": params['downsamplers'], 'lr': args.proj_lr},
        {"params": params['adapter'], "lr": args.proj_lr},
        {"params": params['cross_attn'], "lr": args.proj_lr},
        {"params": params['ctc_head'], "lr": args.proj_lr},
        {"params": params['audio_head'], "lr": args.proj_lr},
        {"params": params['duration_token_params'], "lr": args.proj_lr},
        {"params": params['layer_weights'], "lr": args.proj_lr}
    ] if len(group["params"]) > 0
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=training_args.weight_decay
    )

    custom_optimizers = (optimizer, None)

    if training_args.group_by_length:
        dataset['train'] = dataset['train'].map(compute_length, num_proc=4)
        dataset['eval'] = dataset['eval'].map(compute_length, num_proc=4)

    np.random.seed(0)
    compute_metrics = wrap_compute_metrics(tokenizer, dataset, writer, info)
    compute_metrics_ = compute_metrics if info['do_compute'] else None

    trainer = MultiLossTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        compute_metrics=compute_metrics_,
        optimizers=custom_optimizers,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    if load:
        trainer.remove_callback(TensorBoardCallback)
        trainer.add_callback(OffsetTensorBoardCallback(step_offset=max_step))
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=checkpoint_dir)

    trainer.save_model()
    model.language_tokenizer.save_pretrained(checkpoint_path)

    return checkpoint_path

def get_bad_folder_path(conf):
    bad_folder_path = os.path.join(os.path.expanduser('~'),
                                   conf.dataset_path,
                                   conf.language,
                                   'bad_folder')

    return bad_folder_path

def setup(info, restart: bool = False, device: str = 'cuda', local_rank: int = -1, exp: int = 0):
    conf, args, training_args, date, checkpoint_path, checkpoint_dir, writer = init(info, restart, exp)

    bad_folder = get_bad_folder_path(conf)

    print('='*60)
    print('Experiment Arguments')
    print(json.dumps(args.__dict__, indent=4, default=str))
    print('=' * 60)

    dataset = make_data_module(DATASETS,
                               bad_folder,
                               filters=FILTERS,
                               micro_data=args.micro_data,
                               train_samples=args.micro_size,
                               eval_samples=info['iters'],
                               do_interleave=info['interleave'],
                               temperature=args.interleave_temperature,
                               randomize=args.randomize,
                               normalize_type=args.normalize,
                               exp=exp)

    init_wandb(local_rank, args, info, date, info['model_name'], DATASETS, exp)

    if args.two_stage:
        two_stage_train(dataset, args, training_args, info, checkpoint_path, checkpoint_dir, writer, device, exp=exp)
    else:
        train_model(dataset, args, training_args, info, checkpoint_path, checkpoint_dir, writer, device, exp=exp)


def train(exp: int = 0):
    local_rank, device = init_gpu()

    info = init_info(SPEECH_ENCODER_ID, LANGUAGE_MODEL_ID, DATASETS, MACHINE, DATETIME, INTERLEAVE, ITERS)

    try:
        setup(info, restart=RESTART, device=device, local_rank=local_rank, exp=exp)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0, help='Path to config file')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPUs to be used')
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=['common_voice', 'fleurs', 'hparl', 'tedx', 'logotypographia'], help='datasets')
    parser.add_argument('--iters', type=int, default=800, help='validation samples per dataset')
    parser.add_argument('--restart', default=True, help='Restart from the beginning', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--machine', type=str, default=None, help='machine name of model to load')
    parser.add_argument('--datetime', type=str, default=None, help='datetime of model to load')
    parser.add_argument('--skip_stage1', default=False, help='skip stage 1 training', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--interleave', default=True, help='interleave data', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--attn_implementation', type=str, default='sdpa', help='attention implementation type')
    parser.add_argument('--speech_encoder_id', type=str, default='openai/whisper-large-v3', help='speech encoder id')
    parser.add_argument('--language_model_id', type=str, default='ilsp/Llama-Krikri-8B-Instruct', help='language model id')
    parser.add_argument('--filters', type=str, nargs='+', default=['duration', 'length', 'text'])

    args, unknown = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    SPEECH_ENCODER_ID = args.speech_encoder_id
    LANGUAGE_MODEL_ID = args.language_model_id
    DATASETS = args.datasets
    ITERS = args.iters
    RESTART = args.restart
    MACHINE = args.machine
    DATETIME = args.datetime
    INTERLEAVE = args.interleave
    SKIP_STAGE1 = args.skip_stage1
    ATTN_IMPL = args.attn_implementation
    FILTERS = args.filters

    print('EXPERIMENT SETUP: ')
    print('speech encoder id: ', SPEECH_ENCODER_ID)
    print('language model id: ', LANGUAGE_MODEL_ID)
    print('datasets: ', DATASETS)
    print('filters: ', FILTERS)
    print('iters: ', ITERS)
    print('restart: ', RESTART)
    print('machine: ', MACHINE)
    print('datetime: ', DATETIME)
    print('interleave: ', INTERLEAVE)
    print('skip_stage1: ', SKIP_STAGE1)
    print('attention_implementation: ', ATTN_IMPL)

    train(args.exp)
