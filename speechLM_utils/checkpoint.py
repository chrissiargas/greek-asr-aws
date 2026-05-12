from typing import Dict, List
from datetime import datetime

from numpy.f2py.auxfuncs import throw_error
from torch.utils.tensorboard import SummaryWriter
from os.path import exists, join, isdir
import os
import socket
import json

def get_max_step(checkpoint_dir):
    max_step = 0
    for filename in os.listdir(checkpoint_dir):
        if isdir(join(checkpoint_dir, filename)) and filename.startswith(
            "checkpoint"
        ):
            max_step = max(max_step, int(filename.replace("checkpoint-", "")))

    return max_step

def get_min_step(checkpoint_dir):
    min_step = 1000000000
    for filename in os.listdir(checkpoint_dir):
        if isdir(join(checkpoint_dir, filename)) and filename.startswith(
            "checkpoint"
        ):
            min_step = min(min_step, int(filename.replace("checkpoint-", "")))

    return min_step

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return checkpoint_dir, True

        max_step = get_max_step(checkpoint_dir)
        if max_step == 0:
            return None, is_completed

        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")

        return checkpoint_dir, is_completed

    return None, False

def get_last_name(info: Dict, restart: bool):
    lora = 'LoRA_1'
    dataset = info['train_dataset']
    date = '' if info['datetime'] is None else info['datetime']

    if restart:
        date = datetime.now().strftime("%b%d_%H-%M")

    return f'{lora}_{dataset}@{date}', date

def get_checkpoint(checkpoints_path: str, info: Dict, model_name: str, restart: bool = False):
    load = not restart
    last_name, date = get_last_name(info, restart)

    if 'machine' in info and info['machine'] is not None:
        machine = info['machine']
    else:
        machine = socket.gethostname()

    checkpoint_path = os.path.join(os.path.expanduser('~'),
                                   checkpoints_path,
                                   info['checkpoint_folder'],
                                   machine,
                                   model_name,
                                   last_name)

    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_path)

    if load:
        config_file = join(checkpoint_path, 'config.json')
        if os.path.exists(config_file):
            print(f"Loading model configuration strictly from {config_file}...")
            with open(config_file, 'r') as f:
                args = json.load(f)
        else:
            print(f"WARNING: args.json not found in {checkpoint_path}. Falling back to codebase YAML.")
            args = None

        if 'stage' in info and info['stage'] is not None:
            checkpoint_path = join(checkpoint_path, f"stage{info['stage']}")

        if 'turn' in info and info['turn'] is not None:
            checkpoint_dir = join(checkpoint_path, f"checkpoint-{info['turn']}")
        else:
            min_step = get_min_step(checkpoint_path)
            checkpoint_dir = join(checkpoint_path, f"checkpoint-{min_step}")
            info['turn'] = min_step

        if checkpoint_dir is not None:
            print(f"Found checkpoint at: {checkpoint_dir}")

    else:
        checkpoint_dir = None
        args = None

    return checkpoint_path, checkpoint_dir, writer, date, args