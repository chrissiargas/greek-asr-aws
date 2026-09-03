import os
import sys
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import matplotlib.pyplot as plt
from speechLM_utils.downsamplers import ReshapeAdapter


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class CrossAttention(nn.Module):
    def __init__(self,
                 hidden_dim,
                 audio_dim,
                 audio_offset=0,
                 num_heads=32,
                 causal_fusion: bool = False,
                 downsampler: Optional[nn.Module] = None,
                 downsample_L: int = 5,
                 gated: bool = False,
                 seq_len: int = 300,
                 positional_info: bool = False):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.gated = gated
        self.downsample_L = downsample_L
        self.positional_info = positional_info

        if self.head_dim * num_heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        audio_embed_dim = audio_dim * self.downsample_L if isinstance(downsampler, ReshapeAdapter) else audio_dim

        self.k_proj = nn.Linear(audio_embed_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(audio_embed_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.causal_fusion = causal_fusion
        self.audio_offset = audio_offset

        if self.gated:
            self.gate = nn.Parameter(torch.zeros(1))

        if self.positional_info:
            self.audio_pos_embed = SinusoidalPositionalEmbedding(d_model=audio_embed_dim, max_len=seq_len)

        self.cached_key = None
        self.cached_value = None
        self.cached_mask = None

    def clear_cache(self):
        self.cached_key = None
        self.cached_value = None
        self.cached_mask = None

    def compute_mask(self, prm_len, prm_audio_len, inj_audio_len, device, dtype, prm_audio_mask=None,
                     inj_audio_mask=None, verbose: bool = False):
        min_val = torch.finfo(dtype).min

        if verbose:
            if prm_audio_mask is not None:
                plt.imshow(prm_audio_mask[0].unsqueeze(0).detach().cpu(), aspect='auto')
                plt.show()

            if inj_audio_mask is not None:
                plt.imshow(inj_audio_mask[0].unsqueeze(0).detach().cpu(), aspect='auto')
                plt.show()

        if self.causal_fusion:
            batch_size = inj_audio_mask.shape[0]
            inj_audio_end = inj_audio_mask.sum(dim=1)
            inj_audio_masked = inj_audio_len - inj_audio_end
            prm_audio_end = prm_audio_mask.sum(dim=1)
            prm_audio_masked = prm_audio_len - prm_audio_end

            prm_indices = torch.arange(prm_len, device=device).view(1, prm_len, 1)
            inj_indices = torch.arange(inj_audio_len, device=device).view(1, 1, inj_audio_len)

            content_prm_audio_len = prm_audio_len - prm_audio_masked
            content_inj_audio_len = inj_audio_len - inj_audio_masked

            slope = (content_inj_audio_len / content_prm_audio_len).view(batch_size, 1, 1)

            t_rel = torch.clamp(prm_indices - self.audio_offset, min=0)
            inj_boundary = torch.floor(t_rel * slope)

            causal_mask = torch.where(inj_indices <= inj_boundary, 0.0, min_val)
            causal_mask = causal_mask.unsqueeze(1)

            if verbose and isinstance(causal_mask, torch.Tensor):
                plt.imshow(~causal_mask[0, 0].to(torch.bool).detach().cpu(), aspect='auto', interpolation='nearest')
                plt.show()
        else:
            causal_mask = 0

        if inj_audio_mask is not None:
            audio_mask = inj_audio_mask[:, None, None, :]
            audio_mask = (1 - audio_mask) * min_val
        else:
            audio_mask = 0

        mask = causal_mask + audio_mask

        if verbose and isinstance(causal_mask, torch.Tensor):
            plt.imshow(~mask[0, 0].to(torch.bool).detach().cpu(), aspect='auto', interpolation='nearest')
            plt.show()

        if isinstance(mask, torch.Tensor):
            mask = torch.clamp(mask, min=min_val).to(dtype)

        return mask

    def forward(self, hidden_states, audio_features, prompt_audio=None, inj_audio_mask=None, prm_audio_mask=None):
        batch_size, text_len, _ = hidden_states.shape
        audio_batch_size = audio_features.shape[0]

        # FIX: Dynamically expand acoustic features during beam search generation
        if audio_batch_size != batch_size:
            num_beams = batch_size // audio_batch_size
            audio_features = audio_features.repeat_interleave(num_beams, dim=0)

            if prompt_audio is not None:
                prompt_audio = prompt_audio.repeat_interleave(num_beams, dim=0)
            if inj_audio_mask is not None:
                inj_audio_mask = inj_audio_mask.repeat_interleave(num_beams, dim=0)
            if prm_audio_mask is not None:
                prm_audio_mask = prm_audio_mask.repeat_interleave(num_beams, dim=0)

        inj_audio_len = audio_features.shape[1]
        prm_len = hidden_states.shape[1]

        if prompt_audio is not None:
            prm_audio_len = prompt_audio.shape[1]
        else:
            prm_audio_len = 0

        query = self.q_proj(self.layer_norm(hidden_states))
        query = query.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)

        is_decoding = (text_len == 1)

        if is_decoding and self.cached_key is not None:
            key = self.cached_key
            value = self.cached_value
            mask = self.cached_mask
        else:
            if self.positional_info:
                audio_features = self.audio_pos_embed(audio_features)

            key = self.k_proj(audio_features)
            value = self.v_proj(audio_features)

            key = key.view(batch_size, inj_audio_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, inj_audio_len, self.num_heads, self.head_dim).transpose(1, 2)

            mask = self.compute_mask(prm_len, prm_audio_len,
                                     inj_audio_len,
                                     hidden_states.device,
                                     hidden_states.dtype,
                                     prm_audio_mask,
                                     inj_audio_mask)

            # Save to memory
            self.cached_key = key
            self.cached_value = value
            self.cached_mask = mask

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        scores = scores + mask

        attentions = F.softmax(scores, dim=-1)
        attentions = self.dropout(attentions)

        if getattr(self, "store_attention", False):
            # Capture the attention matrix at this generation step
            self.attention_map.append(attentions.detach().cpu())

        attn_output = torch.matmul(attentions, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, text_len, self.hidden_dim)

        proj_audio_context = self.out_proj(attn_output)
        if self.gated:
            proj_audio_context = torch.tanh(self.gate) * proj_audio_context

        fused_output = hidden_states + proj_audio_context

        return fused_output

class InjectionLayer(nn.Module):
    def __init__(self, lm_layer, cross_attention_layer):
        super().__init__()

        self.LM_layer = lm_layer
        self.cross_attention_layer = cross_attention_layer

        self.injection_audio = None
        self.injection_audio_mask = None
        self.prompt_audio = None
        self.prompt_audio_mask = None

    def __getattr__(self, name):
        try:
            # First, try to get the attribute normally (from InjectionLayer itself)
            return super().__getattr__(name)
        except AttributeError:
            # If it's missing (like 'attention_type'), pass the request down to the wrapped LM_layer.
            # PyTorch stores registered sub-modules in the _modules dictionary.
            if 'LM_layer' in self._modules:
                return getattr(self.LM_layer, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, hidden_states, *args, **kwargs):
        lm_outputs = self.LM_layer(hidden_states, *args, **kwargs)
        lm_hidden_states = lm_outputs[0]

        fused_hidden_states = self.cross_attention_layer(lm_hidden_states,
                                                           self.injection_audio,
                                                           self.prompt_audio,
                                                           self.injection_audio_mask,
                                                           self.prompt_audio_mask)

        rest = lm_outputs[1:] if len(lm_outputs) > 1 else ()
        if isinstance(rest, torch.Tensor):
            rest = (rest,)

        return (fused_hidden_states,) + rest


if __name__ == "__main__":
    print("=== Testing Causal Masking in CrossAttention ===")

    # 1. Setup Dummy Dimensions for the experiment
    batch_size = 1
    hidden_dim = 4096
    audio_dim = 1280
    audio_offset = 0  # Let's say 2 text tokens appear before the audio prompt
    prm_len = 363  # Total LLM tokens (2 text + 8 audio + 3 generated text)
    prm_audio_len = 300  # Prompt-level audio tokens
    inj_audio_len = 500  # Deep-level injection audio tokens (more heavily downsampled)

    # 2. Instantiate CrossAttention
    cross_attn = CrossAttention(
        hidden_dim=hidden_dim,
        audio_dim=audio_dim,
        audio_offset=audio_offset,
        num_heads=8,  # Arbitrary for this test
        causal_fusion=True,
        downsample_L=3
    )

    # 3. Create dummy masks (assuming no padding for this test)
    device = torch.device('cpu')
    dtype = torch.float32
    prm_audio_mask = torch.ones(batch_size, prm_audio_len, device=device)
    prm_audio_mask[:, 150:] = 0
    inj_audio_mask = torch.ones(batch_size, inj_audio_len, device=device)
    inj_audio_mask[:, 250:] = 0

    # 4. Compute Mask using your class method
    mask = cross_attn.compute_mask(
        prm_len=prm_len,
        prm_audio_len=prm_audio_len,
        inj_audio_len=inj_audio_len,
        device=device,
        dtype=dtype,
        prm_audio_mask=prm_audio_mask,
        inj_audio_mask=inj_audio_mask,
        verbose=True
    )

    # # 5. Format and Print Results
    # print(f"LLM Total Sequence Length: {prm_len}")
    # print(f"Prompt Audio Tokens: {prm_audio_len} (starts at offset {audio_offset})")
    # print(f"Injection Audio Tokens: {inj_audio_len}")
    # print("-" * 50)
    #
    # mask_to_print = mask.squeeze()
    #
    # header = "      " + "".join([f"Inj{i:<4}" for i in range(mask_to_print.shape[1])])
    # print(header)
    # print("      " + "-" * (mask_to_print.shape[1] * 7))
    #
    # for i, row in enumerate(mask_to_print):
    #     # Print 0.0 for unmasked (attend), -inf for masked
    #     # formatting heavily negative numbers to ' -inf' for readability
    #     row_str = " ".join([f"{' 0.0' if val == 0.0 else ' -inf':<6}" for val in row])
    #     print(f"LLM{i:<2} | {row_str}")
    #
    # print("================================================")