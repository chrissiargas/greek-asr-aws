import gc
import torch
import os
import wandb
from speechLM_utils.training_info import Info
from config.parser import Parser
from speechLM_utils.checkpoint import get_checkpoint
import json
from transformers import GenerationConfig, Seq2SeqTrainingArguments
from typing import Dict, List
import socket
from config.parser import Args

def get_tags(args: Dict, datasets: List[str]):
    tags = []
    tags.append(socket.gethostname())
    tags.append(','.join(datasets))

    ## Regularization & Augmentation Configurations
    if args['blank_training']:
        tags.append('blank_training')
    if args['text_perturbation']:
        tags.append('text_perturbation')

    ## Input Injection Configurations
    if args['include_adapter']:
        tags.append('include_adapter')
    if args['static_projector']:
        tags.append('static_projector')
    tags.append('downsample_K: ' + str(args['downsample_K']))

    ## Cross-Attention Injection Configurations
    tags.append('downsample_L: ' + str(args['downsample_L']))
    tags.append('injection_downsample: ' + str(args['injection_downsample']))
    tags.append('injection_layers: ' + ','.join(str(args['injection_layers'])))
    tags.append('downsamplers: ' + str(args['downsamplers']))
    if args['gated_cross_attention']:
        tags.append('gated_cross_attention')
    if args['causal_fusion']:
        tags.append('causal_fusion')
    if args['positional_info']:
        tags.append('positional_info')
    if args['pyramid_layers']:
        tags.append('pyramid_layers')

    ## LoRA Configurations
    if args['linguistic_lora']:
        tags.append('linguistic_lora')
    if args['two_stage']:
        tags.append('two_stage')

    ## Auxiliary Losses Configurations
    if args['predict_duration']:
        tags.append('predict_duration')
    if args['ctc']:
        tags.append('ctc')
    if args['audio_forecasting']:
        tags.append('audio_forecasting')

    ## Prompt Configurations
    if args['prompt_persona'] is not None:
        persona_len = len(args['prompt_persona'])
        tags.append(f'prompt_persona len: {persona_len}')
    if args['prompt_instruction'] is not None:
        instruct_len = len(args['prompt_instruction'])
        tags.append(f'prompt_instruction: {instruct_len}')

    if args['prompt_verbatim']:
        tags.append('prompt_verbatim')

    return tags


def init_gpu():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"Process launched on GPU: {local_rank}")
    else:
        device = "cuda"
        print("Running in standard mode (DataParallel or Single GPU)")

    return local_rank, device


def init_info(model_type, datasets, machine=None, datetime=None, interleave=True, iters=800):
    info = Info[model_type]
    info['checkpoint_folder'] = f'{model_type}_checkpoints'
    info['train_dataset'] = datasets
    info['machine'] = machine
    info['datetime'] = datetime
    info['interleave'] = interleave
    info['iters'] = iters

    return info


def init(model_type, info, restart=True, exp: int = 0):
    conf = Parser()
    conf.get_args(exp)

    if model_type == 'continuous_fusion':
        args = conf.cont_fuse_args
        training_args = args.training_args
    elif model_type == 'slam_asr':
        args = conf.slam_args
        training_args = args.training_args
    elif model_type == 'dual_fusion':
        args = conf.dual_fuse_args
        training_args = args.training_args

    if info['model_name'] is None:
        model_name = (info['speech_encoder_id'].split('/')[1] + '_' +
                      info['language_model_id'].split('/')[1])
    else:
        model_name = info['model_name']

    info['model_name'] = model_name

    checkpoint_path, checkpoint_dir, writer, date, loaded_args = get_checkpoint(args.checkpoint_path, info, model_name, restart)

    if loaded_args is not None:
        args = loaded_args
        restored_args = Args()
        for key, value in args.items():
            setattr(restored_args, key, value)
        args = restored_args
    else:
        config_file = os.path.join(checkpoint_path, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(args.__dict__, f)

    training_args['output_dir'] = checkpoint_path
    training_args['logging_dir'] = checkpoint_path
    training_args['report_to'] = ['wandb']
    training_args['generation_config'] = GenerationConfig(**training_args['generation_config'])
    training_args = Seq2SeqTrainingArguments(**training_args)

    bad_folder = os.path.join(os.path.expanduser('~'),
                              conf.dataset_path,
                              conf.language,
                              'bad_folder')

    return conf, args, training_args, bad_folder, date, checkpoint_path, checkpoint_dir, writer


def resume_wandb(local_rank, info):
    if local_rank in [-1, 0]:
        api = wandb.Api()
        runs = api.runs("chrissiargas-innoetics/Greek-ASR",
                        filters={"display_name": info['datetime']})

        if len(runs) > 0:
            run_id = runs[0].id
            print(f"Found existing W&B run '{info['datetime']}' with ID {run_id}. Resuming...")
            wandb.init(project="Greek-ASR", entity="chrissiargas-innoetics", id=run_id, resume="must")
        else:
            print(f"Could not find existing run '{info['datetime']}'. Starting a new evaluation run...")
            wandb.init(project="Greek-ASR", entity="chrissiargas-innoetics", name=f"{info['datetime']}_EVAL",
                       group=info['model_type'])


def init_wandb(local_rank, args, date, model_type, datasets, note):
    tags = get_tags(args, datasets)
    if local_rank in [-1, 0]:
        wandb.init(
            entity="chrissiargas-innoetics",
            project="Greek-ASR",
            name=date,
            group=model_type,
            config=args.__dict__,
            tags=tags,
            notes=note,
            reinit=True
        )