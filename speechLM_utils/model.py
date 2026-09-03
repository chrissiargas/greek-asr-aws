from typing import Dict

from speechLM_utils.dual_fusion_model import DualFusionModel
import torch
from speechLM_utils.checkpoint import get_max_step
import os
import torch.nn as nn
from os.path import join
import json
import safetensors

def load_weights(model: nn.Module, checkpoint_path: str, args: Dict, device: str = 'cuda'):
    max_step = get_max_step(checkpoint_path)
    weights_checkpoint = join(checkpoint_path, f"checkpoint-{max_step}")
    print('Found checkpoint to load: ', weights_checkpoint)

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

        trained_keys = [
            'adapter', 'input_downsampler', 'injection_downsampler', 'injection_downsamplers',
            'cross_attention_layer', 'ctc_predictor', 'audio_predictor', 'duration_predictor'
        ]

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if any(trained_key in k for trained_key in trained_keys):
                new_k = k
                if 'language_model.model.layers' in k:
                    new_k = k.replace('language_model.model.layers', 'language_model.base_model.model.model.layers')
                filtered_state_dict[new_k] = v

        if not filtered_state_dict:
            print(f"[{device}] WARNING: No projector/adapter keys found in Stage 1 checkpoint!")

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected_keys)

        missing_cross_attn = [k for k in missing_keys if 'cross_attention' in k]
        if missing_cross_attn:
            print(f"[{device}] CRITICAL WARNING: Failed to load cross-attention weights!")

        print(f"[{device}] SUCCESS: Stage 1 weights loaded.")
        print(f"[{device}] Loaded {len(filtered_state_dict)} tensors.")

    except Exception as e:
        print(f"[{device}] Error: Could not load weights. {e}")

    return model, max_step

def get_model(args: Dict, info: Dict, device = 'cuda', exp: int = 0, attn_implementation: str = 'sdpa'):

    model = DualFusionModel(
        speech_encoder_model_id=info['speech_encoder_id'],
        language_model_id=info['language_model_id'],
        blank_training=args['blank_training'],
        audio_dropout=args['audio_dropout'],
        text_perturbation=args['text_perturbation'],
        text_dropout=args['text_dropout'],
        spec_augment=args['spec_augment'],
        include_adapter=args['include_adapter'],
        input_downsample=args['input_downsample'],
        static_projector=args['static_projector'],
        downsample_K=args['downsample_K'],
        hidden_dim=args['hidden_dim'],
        static_injection=args['static_injection_layers'],
        injection_layers=args['injection_layers'],
        pyramid_layers=args['pyramid_layers'],
        gated=args['gated_cross_attention'],
        downsample_L=args['downsample_L'],
        injection_downsample=args['injection_downsample'],
        downsamplers=args['downsamplers'],
        causal_fusion=args['causal_fusion'],
        positional_info=args['positional_info'],
        layer_wise_fusion=args['layer_wise_fusion'],
        layer_weights_static=args['layer_weights_static'],
        predict_duration=args['predict_duration'],
        duration_resolution=args['duration_resolution'],
        max_duration=args['max_duration'],
        ctc=args['ctc'],
        audio_forecasting=args['audio_forecasting'],
        ctc_weight=args['ctc_weight'],
        audio_weight=args['audio_weight'],
        duration_weight=args['duration_weight'],
        lng_lora=args['linguistic_lora'],
        lora_r=args['lora_r'],
        acoustic_lora=args['acoustic_lora'],
        lora_params=args['lora_params'],
        prompt_persona=args['prompt_persona'],
        prompt_instruction=args['prompt_instruction'],
        prompt_verbatim=args['prompt_verbatim'],
        dtype=torch.float32,
        bit4=True,
        attn_implementation=attn_implementation,
        device=device,
        exp=exp
    )

    tokenizer = model.language_tokenizer

    return model, tokenizer
