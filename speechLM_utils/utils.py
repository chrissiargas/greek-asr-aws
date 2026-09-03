import gc
import torch
import os
import wandb
from config.parser import Parser
from speechLM_utils.checkpoint import get_checkpoint
import json
from transformers import GenerationConfig, Seq2SeqTrainingArguments
from typing import Dict, List
import socket

def tags_to_str(tags: List[str]):
    return '\n'.join(tags)

def get_tags(args: Dict, info: Dict, datasets: List[str]):
    tags = []
    tags.append(socket.gethostname())

    tags.append(f"Speech Encoder: {info['speech_encoder_id']}")
    tags.append(f"Language Model: {info['language_model_id']}")
    tags.append(f"Interleave Factor: {info['interleave']}")

    ds_string = ','.join(datasets)
    if len(ds_string) > 64:
        tags.append(", ".join([dataset[:5] for dataset in datasets]))
    else:
        tags.append(ds_string)

    ## Regularization & Augmentation Configurations
    if args['text_perturbation']:
        tags.append('text_perturbation')
        tags.append('text_dropout: ' + str(args['text_dropout']))

    ## Input Injection Configurations
    if args['include_adapter']:
        tags.append('include_adapter')
        tags.append('downsample_K: ' + str(args['downsample_K']))

    ## Cross-Attention Injection Configurations
    tags.append('downsample_L: ' + str(args['downsample_L']))
    tags.append('injection_layers: ' + ', '.join(args['injection_layers']))
    if args['gated_cross_attention']:
        tags.append('gated_cross_attention')
    if args['causal_fusion']:
        tags.append('causal_fusion')
    if args['positional_info']:
        tags.append('positional_info')
    if args['pyramid_layers']:
        tags.append('pyramid_layers')
    if args['layer_wise_fusion']:
        tags.append('layer_wise_fusion')

    ## LoRA Configurations
    if args['linguistic_lora']:
        tags.append('linguistic_lora')
        tags.append('lora_r: ' + str(args['lora_r']))
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


def init_info(speech_encoder_id, language_model_id, datasets, machine=None, datetime=None, interleave=True, iters=800):
    model_name = (speech_encoder_id.split('/')[1] + '_' + language_model_id.split('/')[1])

    info = {
        'checkpoint_folder': 'dual_fusion_checkpoints',
        'model_name': model_name,
        'train_dataset': datasets,
        'interleave': interleave,
        'speech_encoder_id': speech_encoder_id,
        'language_model_id': language_model_id,
        'machine': machine,
        'datetime': datetime,
        'iters': iters,
        'do_compute': True,
        'compute_wer_cer': True,
        'N_samples_for_metrics': None

    }

    return info


def init(info, restart=True, exp: int = 0):
    conf = Parser()
    conf.get_args(exp)

    args = conf.dual_fuse_args
    training_args = args.training_args

    checkpoint_path, checkpoint_dir, writer, date, loaded_args = get_checkpoint(args.checkpoint_path,
                                                                                info,
                                                                                restart)

    if loaded_args is not None:
        args.__dict__.update(loaded_args)
        training_args = args.training_args

    else:
        config_file = os.path.join(checkpoint_path, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(args.__dict__, f)

    training_args['output_dir'] = checkpoint_path
    training_args['logging_dir'] = checkpoint_path
    training_args['report_to'] = ['wandb']
    training_args['generation_config'] = GenerationConfig(**training_args['generation_config'])
    training_args = Seq2SeqTrainingArguments(**training_args)

    return conf, args, training_args, date, checkpoint_path, checkpoint_dir, writer


def resume_wandb(local_rank, date, model_name, exp):
    id = f'{date}-Exp:{exp}'
    if local_rank in [-1, 0]:
        api = wandb.Api()
        runs = api.runs("chrissiargas-innoetics/Hungarian-ASR",
                        filters={"display_name": id})

        if len(runs) > 0:
            run_id = runs[0].id
            print(f"Found existing W&B run '{id}' with ID {run_id}. Resuming...")
            wandb.init(project="Hungarian-ASR",
                       entity="chrissiargas-innoetics",
                       id=run_id, resume="must")
        else:
            print(f"Could not find existing run '{id}'. Starting a new evaluation run...")
            wandb.init(project="Hungarian-ASR",
                       entity="chrissiargas-innoetics",
                       name=f"{id}_Eval",
                       group=model_name)


def init_wandb(local_rank, args, info, date, model_name, datasets, exp):
    tags = get_tags(args, info, datasets)
    note = tags_to_str(tags)

    if local_rank in [-1, 0]:
        wandb.init(
            entity="chrissiargas-innoetics",
            project="Hungarian-ASR",
            name=f'{date}-Exp:{exp}',
            group=model_name,
            config=args.__dict__,
            tags=tags,
            notes=note,
            reinit=True
        )