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
import torchaudio

import torchaudio
from pydub import AudioSegment
from pydub.playback import play

def play_audio_sample(audio_path):
    print(f"Playing: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    play(audio)

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

def run_custom_task(info, device='cuda'):
    conf, args, _, bad_folder, _, checkpoint_path, checkpoint_dir, _ = init(info['model_type'], info, restart=False)

    print(f"Loading Model ...")

    model = load_model(info['model_type'], args, info, checkpoint_path, checkpoint_dir, device)
    tokenizer = model.language_tokenizer

    raw_audio, sample_rate = torchaudio.load(info['audio_file'])
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        raw_audio = resampler(raw_audio)
        raw_audio = raw_audio.squeeze().numpy()

    input = model.processor(
        raw_audio,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True
    )

    audio = input.input_features.to(device)
    audio_mask = input.attention_mask.to(device)

    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        num_beams=4
    )

    # 1. Update the internal instruction
    model.prompt_instruction = info['prompt']

    # 2. Refresh the embedding bank to generate new tensors for this prompt
    model.set_embed_bank()

    # 3. Execution
    print(f"Executing Task: {info['prompt']}")

    with torch.no_grad():
        output_ids = model.generate(
            audios=audio,
            audio_masks=audio_mask,
            generation_config=gen_config
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("-" * 30)
    print(f"Model Response: {response}")
    print("-" * 30)


FILTERS = ['duration', 'length']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type=str, default=None, help='audio file to be processed')
    parser.add_argument('--prompt', type=str, default=None, help='prompt to be asked')
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

    try:
        base_info['audio_file'] = args.audio_file
        base_info['prompt'] = args.prompt

        run_custom_task(base_info, device='cuda')

    finally:
        pass