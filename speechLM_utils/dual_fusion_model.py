import contextlib
import sys
from os.path import dirname, abspath

from mcp.cli.cli import dev
from pyarrow import duration

sys.path.insert(0, dirname(dirname(abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

from speechLM_utils.environment import set_environment
set_environment()

import numpy
import torch
import torch.nn as nn
from torchmetrics.text import CharErrorRate, WordErrorRate
from transformers import (AutoModelForCausalLM, AutoTokenizer, WhisperProcessor,
                          BitsAndBytesConfig, AutoModel, AutoProcessor, AutoConfig, WhisperForConditionalGeneration)
from transformers import WhisperModel

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from config.parser import Parser
import gc
from speechLM_utils.continuous_fusion_utils import CrossAttention, InjectionLayer
import torchaudio.transforms as T
from speechLM_utils.downsamplers import get_downsampler, CIFireAdapter
from typing import Optional
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
val_num_beams = 2
stopping_criteria = None
cer = CharErrorRate()
wer = WordErrorRate()
access_token = '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'
from speechLM_utils.data_collator import NUM_CLASSES, BLANK_IDX
import torch.nn.functional as F

def apply_text_dropout(batch_labels, batch_label_masks, tokenizer, dropout_prob=0.1, device=None):
    noisy_labels = batch_labels.clone()

    if device is None:
        device = batch_labels.device

    batch_size, seq_len = batch_labels.shape
    rand_matrix = torch.rand(batch_size, seq_len, device=device)
    valid_mask = (batch_labels != tokenizer.pad_token_id) & \
                 (batch_labels != tokenizer.eos_token_id) & \
                 (batch_label_masks == 1)

    mask_indices = (rand_matrix < dropout_prob) & valid_mask
    # random_tokens = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=batch_labels.device)
    noisy_labels[mask_indices] = tokenizer.pad_token_id

    return noisy_labels

def apply_audio_dropout(batch_audios, batch_labels, batch_label_masks, tokenizer, dropout_prob=0.05):
    batch_size = batch_audios.shape[0]
    drop_indices = torch.rand(batch_size) < dropout_prob

    noisy_audios = batch_audios.clone()
    new_labels = batch_labels.clone()
    new_masks = batch_label_masks.clone() if batch_label_masks is not None else None

    if not drop_indices.any():
        return batch_audios, batch_labels, batch_label_masks

    noise = torch.randn_like(batch_audios[drop_indices])
    noisy_audios[drop_indices] = noise

    new_labels[drop_indices, :] = tokenizer.pad_token_id
    new_labels[drop_indices, 0] = tokenizer.eos_token_id

    if batch_label_masks is not None:
        new_masks[drop_indices, :] = 0
        new_masks[drop_indices, 0] = 1

    return noisy_audios, new_labels, new_masks

@dataclass
class DualFusionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    loss_ctc: Optional[torch.FloatTensor] = None
    loss_audio: Optional[torch.FloatTensor] = None
    loss_duration: Optional[torch.FloatTensor] = None

class DualFusionModel(nn.Module):
    def __init__(self,
                 speech_encoder_model_id,
                 language_model_id,
                 blank_training,
                 audio_dropout,
                 text_perturbation,
                 text_dropout,
                 spec_augment,
                 include_adapter,
                 input_downsample,
                 static_projector,
                 downsample_K,
                 hidden_dim,
                 static_injection,
                 injection_layers,
                 pyramid_layers,
                 gated,
                 downsample_L,
                 injection_downsample,
                 downsamplers,
                 causal_fusion,
                 positional_info,
                 predict_duration,
                 duration_resolution,
                 max_duration,
                 ctc,
                 audio_forecasting,
                 ctc_weight,
                 duration_weight,
                 audio_weight,
                 lng_lora,
                 acoustic_lora,
                 lora_params,
                 prompt_persona,
                 prompt_instruction,
                 prompt_verbatim,
                 dtype,
                 bit4,
                 attn_implementation,
                 device,
                 exp):

        super().__init__()
        super(DualFusionModel, self).__init__()

        self.conf = Parser()
        self.conf.get_args(exp)

        self.blank_training = blank_training
        self.audio_dropout = audio_dropout
        self.text_perturbation = text_perturbation
        self.text_dropout = text_dropout
        self.spec_augment = spec_augment

        self.include_adapter = include_adapter
        self.static_projector = static_projector
        self.downsample_K = downsample_K
        self.hidden_dim = hidden_dim
        self.input_downsample = input_downsample

        self.static_injection = static_injection
        self.injection_layer_ids = injection_layers
        self.downsample_L = downsample_L
        self.causal_fusion = causal_fusion
        self.injection_downsample = injection_downsample
        self.downsamplers = downsamplers
        self.pyramid_layers = pyramid_layers
        self.gated = gated
        self.positional_info = positional_info

        self.lng_lora = lng_lora
        self.acoustic_lora = acoustic_lora
        self.lora_params = lora_params

        self.predict_duration = predict_duration
        self.ctc = ctc
        self.audio_forecasting = audio_forecasting
        self.text_offset = 0
        self.duration_resolution = duration_resolution
        self.max_duration = max_duration

        self.n_injections = len(self.injection_layer_ids)
        self.prompt_persona = prompt_persona
        self.prompt_instruction = prompt_instruction
        self.prompt_verbatim = prompt_verbatim

        self.device = device
        self.dtype = dtype

        if bit4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        lm_config = AutoConfig.from_pretrained(language_model_id)
        language_project_dim = lm_config.hidden_size

        self.processor = WhisperProcessor.from_pretrained(speech_encoder_model_id,
                                                          language='el',
                                                          task='transcribe',
                                                          predict_timestamps=False)
        if self.spec_augment:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=27)
            self.time_masking = T.TimeMasking(time_mask_param=100)

        self.speech_model = WhisperModel.from_pretrained(speech_encoder_model_id, torch_dtype=dtype).to(self.device)
        self.speech_encoder = self.speech_model.encoder
        self.device = self.speech_encoder.device

        self.audio_dim = int(self.speech_model.config.d_model)
        self.audio_len = 1500

        del self.speech_model
        gc.collect()
        torch.cuda.empty_cache()

        if self.acoustic_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=self.lora_params,
                lora_dropout=0.1,
                bias='none'
            )

            self.speech_encoder = get_peft_model(self.speech_encoder, peft_config)

        else:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False

        if self.include_adapter:
            self.input_downsampler, self.downsampled_dim = get_downsampler(self.input_downsample,
                                                                           self.downsample_K,
                                                                           self.audio_dim,
                                                                           self.device,
                                                                           self.dtype)

            self.adapter = nn.Sequential(
                nn.Linear(self.downsampled_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, language_project_dim),
                nn.LayerNorm(language_project_dim)
            ).to(self.device, dtype=self.speech_encoder.dtype)

        self.language_tokenizer = AutoTokenizer.from_pretrained(
            language_model_id,
            trust_remote_code=True,
            token=access_token
        )

        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        if self.predict_duration:
            num_bins = int(self.max_duration / self.duration_resolution) + 1

        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            token=access_token
        ).to(device=self.device)

        if bit4:
            self.language_model = prepare_model_for_kbit_training(
                self.language_model,
                use_gradient_checkpointing=True
            )

        if self.lng_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=self.lora_params,
                bias='none',
                modules_to_save=[],
            )
            self.language_model = get_peft_model(self.language_model, peft_config)

        else:
            for param in self.language_model.parameters():
                param.requires_grad = False
            self.language_model.eval()

        self.embed_bank = {"embed1": None, "embed2": None, "att1": None, "att2": None}

        self.set_embed_bank()
        self.audio_offset = self.embed_bank['embed1'].shape[1]

        if self.lng_lora:
            layers = self.language_model.get_base_model().model.layers
        else:
            layers = self.language_model.model.layers

        self.injection_downsampler = None

        if self.downsamplers == 'common' and self.downsample_L == self.downsample_K:
            self.injection_downsampler = self.input_downsampler
        elif self.downsamplers == 'injection_common' and self.downsample_L > 1:
            self.injection_downsampler, _ = get_downsampler(self.injection_downsample,
                                                            self.downsample_L,
                                                            self.audio_dim,
                                                            self.device,
                                                            self.dtype)

        self.injection_downsamplers = []
        if self.downsamplers == 'different':
            for _ in self.injection_layer_ids:
                if self.downsample_L > 1:
                    injection_downsampler, _ = get_downsampler(self.injection_downsample,
                                                                self.downsample_L,
                                                                self.audio_dim,
                                                                self.device,
                                                                self.dtype)
                else:
                    injection_downsampler = None

                self.injection_downsamplers.append(injection_downsampler)

        for l, injection_layer_id in enumerate(self.injection_layer_ids):
            injection_downsampler = self.injection_downsamplers[l] if self.downsamplers == 'different' else self.injection_downsampler

            cross_attn = CrossAttention(
                hidden_dim=language_project_dim,
                audio_dim=self.audio_dim,
                num_heads=32,
                causal_fusion=self.causal_fusion,
                audio_offset=self.audio_offset,
                downsampler=injection_downsampler,
                downsample_L=self.downsample_L,
                gated=self.gated,
                seq_len=self.audio_len // self.downsample_L,
                positional_info=self.positional_info
            ).to(self.device, dtype=dtype)

            injection_layer = layers[injection_layer_id]
            injection_layer_wrapper = InjectionLayer(injection_layer, cross_attn)
            layers[injection_layer_id] = injection_layer_wrapper

        if self.ctc:
            self.ctc_predictor = nn.Linear(language_project_dim, NUM_CLASSES)
            self.ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

        if self.audio_forecasting:
            self.audio_predictor = nn.Sequential(
                nn.Linear(language_project_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.downsampled_dim)
            ).to(self.device, dtype=self.speech_encoder.dtype)
            self.audio_loss = nn.MSELoss(reduction='none')

        if self.predict_duration:
            self.duration_predictor = nn.Linear(
                language_project_dim, num_bins
            ).to(self.device, dtype=self.speech_encoder.dtype)
            self.duration_loss = nn.CrossEntropyLoss()

        self.ctc_weight = ctc_weight
        self.audio_weight = audio_weight
        self.duration_weight = duration_weight
        self.cif_weight = 1.0

        self.get_gradient()
        self.with_adapter_gradient = torch.no_grad() if self.static_projector else contextlib.nullcontext()
        self.with_injection_gradient = torch.no_grad() if self.static_injection else contextlib.nullcontext()

        if self.static_projector and self.include_adapter:
            for param in self.adapter.parameters(): param.requires_grad = False
            for param in self.input_downsampler.parameters(): param.requires_grad = False

        if self.static_injection:
            for layer in self.injection_layers:
                for param in layer.cross_attention_layer.parameters(): param.requires_grad = False
            if self.injection_downsampler:
                for param in self.injection_downsampler.parameters(): param.requires_grad = False
            for ds in self.injection_downsamplers:
                for param in ds.parameters(): param.requires_grad = False

    @property
    def injection_layers(self):
        for layer_id in self.injection_layer_ids:
            yield self.injection_layer(layer_id)

    def injection_layer(self, layer_id):
        if self.lng_lora:
            return self.language_model.get_base_model().model.layers[layer_id]
        else:
            return self.language_model.model.layers[layer_id]

    def get_named_params(self):
        lora_params = []
        down_params = []
        adapter_params = []
        cross_attn_params = []
        ctc_params = []
        audio_params = []
        duration_params = []
        core_params = []

        for name, param in self.named_parameters():
            is_lora = "lora" in name
            is_down = ("input_downsampler" in name or "injection_downsampler" in name or 'injection_downsamplers' in name)
            is_cross_attn = "cross_attention_layer" in name
            is_adapter = "adapter" in name
            is_ctc_head = "ctc_predictor" in name
            is_audio_head = "audio_predictor" in name
            is_duration_token_parameters = "duration_predictor" in name

            is_core = not param.requires_grad

            if is_lora:
                lora_params.append(param)
            elif is_down:
                down_params.append(param)
            elif is_adapter:
                adapter_params.append(param)
            elif is_cross_attn:
                cross_attn_params.append(param)
            elif is_ctc_head:
                ctc_params.append(param)
            elif is_audio_head:
                audio_params.append(param)
            elif is_duration_token_parameters:
                duration_params.append(param)
            elif is_core:
                core_params.append(param)

        return {'lora': lora_params,
                'downsamplers': down_params,
                'adapter': adapter_params,
                'cross_attn': cross_attn_params,
                'ctc_head': ctc_params,
                'audio_head': audio_params,
                'duration_token_params': duration_params,
                'core': core_params}

    def get_gradient(self):
        trainable_params = 0
        all_params = 0

        llm_lora_params = 0
        whisper_lora_params = 0
        input_down_params = 0
        injection_down_params = 0
        adapter_params = 0
        cross_attn_params = 0
        frozen_whisper_params = 0
        frozen_llm_params = 0
        ctc_params = 0
        duration_params = 0
        audio_params = 0

        print("\n---MODEL PARAMETERS---")
        for name, param in self.named_parameters():
            all_params += param.numel()

            is_llm_lora = ("lora" in name and "language_model" in name)
            is_whisper_lora = ("lora" in name and ("speech_model" in name or "speech_encoder" in name or "speech_decoder" in name))
            is_cross_attn = "cross_attention_layer" in name
            is_input_down = "input_downsampler" in name
            is_injection_down = ("injection_downsampler" in name or 'injection_downsamplers' in name)
            is_adapter = "adapter" in name
            is_whisper = ("speech_model" in name or "speech_encoder" in name or "speech_decoder" in name)
            is_llm = "language_model" in name
            is_ctc_head = "ctc_predictor" in name
            is_duration_token_parameters = "duration_predictor" in name
            is_audio_head = "audio_predictor" in name

            if param.requires_grad:
                trainable_params += param.numel()
                if is_llm_lora: llm_lora_params += param.numel()
                if is_whisper_lora: whisper_lora_params += param.numel()
                if is_cross_attn: cross_attn_params += param.numel()
                if is_input_down: input_down_params += param.numel()
                if is_injection_down: injection_down_params += param.numel()
                if is_adapter: adapter_params += param.numel()
                if is_ctc_head: ctc_params += param.numel()
                if is_duration_token_parameters: duration_params += param.numel()
                if is_audio_head: audio_params += param.numel()
            else:
                if is_whisper: frozen_whisper_params += param.numel()
                if is_llm: frozen_llm_params += param.numel()
                if is_input_down: input_down_params += param.numel()
                if is_adapter: adapter_params += param.numel()

        adapter_trainable = 'Frozen' if self.static_projector else 'Trainable'
        injection_trainable = 'Frozen' if self.static_injection else 'Trainable'

        print("-" * 90)
        print(f"TOTAL PARAMS: {all_params:,}")
        print(f"TRAINABLE PARAMS: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%)")
        print("-" * 30)
        print(f" > LLM LoRA Params (Trainable):      {llm_lora_params:,}")
        print(f" > Whisper LoRA Params (Trainable): {whisper_lora_params:,}")
        print(f" > Input Downsampler Params ({adapter_trainable}): {input_down_params:,}")
        print(f" > Adapter Params ({adapter_trainable}): {adapter_params:,}")
        print(f" > Injection Downsampler Params ({injection_trainable}): {injection_down_params:,}")
        print(f" > Cross Attention Params ({injection_trainable}): {cross_attn_params:,}")
        print(f" > CTC Head Params (Trainable): {ctc_params:,}")
        print(f" > Audio Head Params (Trainable): {audio_params:,}")
        print(f" > Duration Token Params (Trainable): {duration_params:,}")
        print(f" > Base LLM (Frozen):            {frozen_llm_params:,}")
        print(f" > Whisper (Frozen):     {frozen_whisper_params:,}")
        print("=" * 50 + "\n")

        print("\n--- Module Parameters ---")
        encoder_params = 0
        cross_attn_params = 0
        input_down_params = 0
        injection_down_params = 0
        adapter_params = 0
        llm_params = 0
        ctc_params = 0
        audio_params = 0
        duration_params = 0

        for param in self.speech_encoder.parameters():
            encoder_params += param.numel()

        for injection_layer in self.injection_layers:
            for param in injection_layer.cross_attention_layer.parameters():
                cross_attn_params += param.numel()

        if self.include_adapter:
            for param in self.input_downsampler.parameters():
                input_down_params += param.numel()

        if self.downsamplers == 'injection_common' and self.downsample_L > 1:
            for param in self.injection_downsampler.parameters():
                injection_down_params += param.numel()

        elif self.downsamplers == 'different' and self.downsample_L > 1:
            for injection_downsampler in self.injection_downsamplers:
                for param in injection_downsampler.parameters():
                    injection_down_params += param.numel()

        if self.include_adapter:
            for param in self.adapter.parameters():
                adapter_params += param.numel()

        if self.ctc:
            for param in self.ctc_predictor.parameters():
                ctc_params += param.numel()

        if self.audio_forecasting:
            for param in self.audio_predictor.parameters():
                audio_params += param.numel()

        if self.predict_duration:
            for param in self.duration_predictor.parameters():
                duration_params += param.numel()

        for param in self.language_model.parameters():
            llm_params += param.numel()

        print(f"Input Downsampler Params: {input_down_params:,}")
        print(f"Adapter Params: {adapter_params:,}")
        print(f"Injection Downsampler Params: {injection_down_params:,}")
        print(f"Cross-Attention Params: {cross_attn_params:,}")
        print(f"CTC Head Params: {ctc_params:,}")
        print(f"Audio Head Params: {audio_params:,}")
        print(f"Duration Params: {duration_params:,}")
        print(f"Whisper Params: {encoder_params:,}")
        print(f"LLM Params: {llm_params:,}")
        print("----------------------------\n")

    def gradient_checkpointing_enable(self, **kwargs):
        self.language_model.gradient_checkpointing_enable(**kwargs)

    def get_text_embeddings(self, text, device):
        tokens = self.language_tokenizer(
            text, return_tensors="pt", padding=False, truncation=True, max_length=1024, add_special_tokens=False
        ).to(device)
        token_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        embeddings = self.get_token_embeddings(token_ids)
        return embeddings, attention_mask

    def get_token_embeddings(self, token_ids):
        return self.language_model.get_input_embeddings()(token_ids)

    def set_embed_bank(self):
        if self.prompt_persona is None or str(self.prompt_persona).strip().lower() == "none":
            msg_pre_audio = [
                {"role": "user", "content": "AudioContentPlaceholder"}
            ]

        else:
            msg_pre_audio = [
                {"role": "user", "content": self.prompt_persona},
                {"role": "user", "content": "AudioContentPlaceholder"}
            ]

        full_prompt = self.language_tokenizer.apply_chat_template(msg_pre_audio, tokenize=False, add_generation_prompt=True)
        self.prompt_part1, prompt_part2 = full_prompt.split("AudioContentPlaceholder")
        self.prompt_part2 = self.prompt_instruction + prompt_part2

        e1, a1 = self.get_text_embeddings([self.prompt_part1], device=self.device)
        e2, a2 = self.get_text_embeddings([self.prompt_part2], device=self.device)

        self.embed_bank["embed1"] = e1.to(dtype=self.dtype)
        self.embed_bank["att1"] = a1
        self.embed_bank["embed2"] = e2.to(dtype=self.dtype)
        self.embed_bank["att2"] = a2

    def _prepare_input_embeds(
            self, batch_size, audio_embeds = None, audio_masks = None,
            label_ids= None, label_masks = None, noisy_label_ids = None,
            tag_tokens = None, tag_masks = None
    ):
        target_dtype = self.embed_bank["embed1"].dtype

        if audio_embeds is not None:
            if audio_embeds.dtype != target_dtype and audio_masks is not None:
                audio_embeds = audio_embeds.to(dtype=target_dtype)

        user_embeds = self.embed_bank["embed1"].to(self.device).repeat(batch_size, 1, 1)
        assistant_embeds = self.embed_bank["embed2"].to(self.device).repeat(batch_size, 1, 1)
        user_mask = self.embed_bank["att1"].to(self.device).repeat(batch_size, 1)
        assistant_mask = self.embed_bank["att2"].to(self.device).repeat(batch_size, 1)

        if tag_tokens is not None:
            tag_embeds = self.get_token_embeddings(tag_tokens.to(self.device))
            tag_masks = tag_masks.to(self.device)
        else:
            tag_embeds = None

        embed_components = [user_embeds]
        mask_components = [user_mask]

        if audio_embeds is not None:
            embed_components.append(audio_embeds)
            mask_components.append(audio_masks)

        embed_components.append(assistant_embeds)
        mask_components.append(assistant_mask)

        if tag_embeds is not None:
            embed_components.append(tag_embeds)
            mask_components.append(tag_masks)

        if label_ids is not None:
            label_ids = label_ids.to(self.device)
            input_context_ids = noisy_label_ids.to(self.device) if noisy_label_ids is not None else label_ids
            label_embeds = self.get_token_embeddings(input_context_ids)

            embed_components.append(label_embeds)
            mask_components.append(label_masks)

        prompt_embed = torch.cat(embed_components, dim=1)
        prompt_mask = torch.cat(mask_components, dim=1)

        if label_ids is not None:
            labels = torch.full(
                (batch_size, prompt_embed.shape[1]),
                -100
            ).to(self.device)

            labels[:, -label_ids.shape[1]:] = torch.where(
                label_masks.bool(), label_ids, torch.full_like(label_ids, -100))

            actual_label_length = (label_masks == 1).sum(dim=1).max().detach()

        else:
            actual_label_length = None
            labels = None

        return prompt_embed, prompt_mask, actual_label_length, labels

    def calculate_mask(self, old_mask, new_embeds):
        stride = old_mask.shape[-1] // new_embeds.shape[1]

        new_mask = F.max_pool1d(
            old_mask.float().unsqueeze(1),
            kernel_size=stride
        ).squeeze(1)

        return new_mask

    def get_input_embeddings(self, audio_embeddings, audio_masks):
        alphas = None
        if isinstance(self.input_downsampler, CIFireAdapter):
            down_embeddings, down_masks, alphas = self.input_downsampler(audio_embeddings, audio_masks)
        else:
            down_embeddings = self.input_downsampler(audio_embeddings)
            down_masks = self.calculate_mask(audio_masks, down_embeddings)

        proj_embeddings = self.adapter(down_embeddings)

        return proj_embeddings, down_embeddings, down_masks, alphas

    def ctc_calculate(self, projections, masks, labels, lengths):
        ctc_logits = self.ctc_predictor(projections)
        ctc_log_probs = nn.functional.log_softmax(ctc_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.transpose(0, 1)

        input_lengths = masks.sum(dim=1).long()

        ctc_loss = self.ctc_loss(
            ctc_log_probs,
            labels,
            input_lengths,
            lengths
        )

        return ctc_loss

    def inject(self, encoder_outputs, audio_embs, audio_masks, down_embs, down_masks):
        injection_audios = []
        injection_masks = []

        if self.n_injections == 0:
            return injection_audios, injection_masks

        if not self.pyramid_layers:
            if self.downsample_L == 1:
                inj_audio_mask = self.calculate_mask(audio_masks, audio_embs)
                injection_audio = audio_embs
            elif self.downsamplers == 'common' and down_embs is not None:
                injection_audio = down_embs
                inj_audio_mask = down_masks
            elif self.downsamplers == 'injection_common':
                if isinstance(self.injection_downsampler, CIFireAdapter):
                    injection_audio, inj_audio_mask, _ = self.injection_downsampler(audio_embs, audio_masks)
                else:
                    injection_audio = self.injection_downsampler(audio_embs)
                    inj_audio_mask = self.calculate_mask(audio_masks, injection_audio)

        for l, injection_layer in enumerate(self.injection_layers):
            if self.pyramid_layers:
                if l == self.n_injections - 1:
                    audio_features = encoder_outputs.last_hidden_state
                else:
                    audio_features = encoder_outputs.hidden_states[injection_layer]

                if self.downsample_L > 1:
                    if isinstance(self.injection_downsamplers[l], CIFireAdapter):
                        injection_audio, inj_audio_mask, _ = self.injection_downsamplers[l](audio_features, audio_masks)
                    else:
                        injection_audio = self.injection_downsamplers[l](audio_features)
                        inj_audio_mask = self.calculate_mask(audio_masks, injection_audio)
                else:
                    inj_audio_mask = self.calculate_mask(audio_masks, audio_features)
                    injection_audio = audio_features

            elif self.downsamplers == 'different':
                if self.downsample_L > 1:
                    if isinstance(self.injection_downsamplers[l], CIFireAdapter):
                        injection_audio, inj_audio_mask, _ = self.injection_downsamplers[l](audio_embs, audio_masks)
                    else:
                        injection_audio = self.injection_downsamplers[l](audio_embs)
                        inj_audio_mask = self.calculate_mask(audio_masks, injection_audio)
                else:
                    inj_audio_mask = self.calculate_mask(audio_masks, audio_embs)
                    injection_audio = audio_embs

            injection_audios.append(injection_audio)
            injection_masks.append(inj_audio_mask)

        return injection_audios, injection_masks

    def get_audio_loss(self, outputs, down_embs, down_masks):
        U = self.embed_bank["embed1"].shape[1]
        A = down_embs.shape[1]
        last_hidden_states = outputs.hidden_states[-1]
        audio_hiddens = last_hidden_states[:, U: U + A - 1, :]

        target_embeddings = down_embs[:, 1:]
        predicted_embeddings = self.audio_predictor(audio_hiddens)

        if down_masks is not None:
            valid_mask = down_masks[:, 1:].unsqueeze(-1).to(self.dtype)
        else:
            valid_mask = torch.ones_like(target_embeddings)

        audio_loss = self.audio_loss(predicted_embeddings, target_embeddings)
        aux_audio_loss = (audio_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        norm_audio_loss = aux_audio_loss / self.downsampled_dim

        return norm_audio_loss

    def get_duration_loss(self, outputs, proj_masks, duration_ids):
        prompt_end_idx = (self.embed_bank['embed1'].shape[1] +
                          proj_masks.shape[1] +
                          self.embed_bank['embed2'].shape[1] - 1)

        last_hidden_states = outputs.hidden_states[-1]
        prompt_hidden = last_hidden_states[:, prompt_end_idx, :]

        duration_logits = self.duration_predictor(prompt_hidden)
        duration_ids = duration_ids.long()
        duration_loss = self.duration_loss(duration_logits, duration_ids)

        return duration_loss

    def forward(self, audios, audio_masks,
                audio_lb_tokens = None,
                labels = None,
                label_lengths = None,
                label_masks = None,
                duration_ids = None,
                index=None,
                ctc_labels=None,
                ctc_lengths=None,
                tag_tokens=None,
                tag_masks=None,
                **kwargs):

        batch_size = audios.shape[0]

        if self.training and self.blank_training:
            audios, labels, label_masks = apply_audio_dropout(audios, labels, label_masks, self.language_tokenizer,
                                                              dropout_prob = self.audio_dropout)

        if self.training and self.spec_augment:
            audios = self.freq_masking(audios)
            audios = self.time_masking(audios)

        if self.training and self.text_perturbation:
            noisy_label_ids = apply_text_dropout(labels, label_masks, self.language_tokenizer,
                                                 dropout_prob = self.text_dropout, device = self.device)
        else:
            noisy_label_ids = None

        with torch.no_grad():
            encoder_outputs = self.speech_encoder(audios, attention_mask=audio_masks, output_hidden_states=True)

        audio_embeddings = encoder_outputs.last_hidden_state
        proj_embeddings = None
        down_embeddings = None
        down_masks = None

        with self.with_adapter_gradient:
            if self.include_adapter:
                (proj_embeddings,
                 down_embeddings,
                 down_masks,
                 alphas) = self.get_input_embeddings(audio_embeddings, audio_masks)

                if self.ctc:
                    aux_ctc_loss = self.ctc_calculate(proj_embeddings, down_masks, ctc_labels, ctc_lengths)

        prompt_embed, prompt_mask, label_length, true_labels = self._prepare_input_embeds(batch_size,
                                                                                        proj_embeddings,
                                                                                        down_masks,
                                                                                        labels,
                                                                                        label_masks,
                                                                                        noisy_label_ids,
                                                                                        tag_tokens,
                                                                                        tag_masks)

        with self.with_injection_gradient:
            injection_audios, injection_masks = self.inject(encoder_outputs,
                                                            audio_embeddings,
                                                            audio_masks,
                                                            down_embeddings,
                                                            down_masks)

        for injection_audio, injection_mask, injection_layer in zip(injection_audios, injection_masks, self.injection_layers):
            injection_layer.injection_audio = injection_audio
            injection_layer.injection_audio_mask = injection_mask
            injection_layer.prompt_audio = proj_embeddings
            injection_layer.prompt_audio_mask = down_masks

        try:
            outputs = self.language_model(
                inputs_embeds=prompt_embed,
                attention_mask=prompt_mask.bool(),
                labels=true_labels,
                output_hidden_states=True,
                **kwargs
            )

            if self.audio_forecasting:
                aux_audio_loss = self.get_audio_loss(outputs, down_embeddings, down_masks)

            if self.predict_duration:
                aux_duration_loss = self.get_duration_loss(outputs, down_masks, duration_ids)

            with_CIF = self.include_adapter and isinstance(self.input_downsampler, CIFireAdapter)
            if with_CIF:
                predicted_lengths = alphas.sum(-1)
                target_lengths = ctc_lengths.float()
                aux_cif_loss = nn.functional.l1_loss(predicted_lengths, target_lengths)

            aux_ctc_loss = aux_ctc_loss if self.ctc else torch.tensor(0.0, device=outputs.loss.device)
            norm_audio_loss = aux_audio_loss if self.audio_forecasting else torch.tensor(0.0, device=outputs.loss.device)
            aux_duration_loss = aux_duration_loss if self.predict_duration else torch.tensor(0.0, device=outputs.loss.device)
            aux_cif_loss = aux_cif_loss if with_CIF else torch.tensor(0.0, device=outputs.loss.device)

            total_loss = (outputs.loss +
                          (aux_ctc_loss * self.ctc_weight) +
                          (norm_audio_loss * self.audio_weight) +
                          (aux_duration_loss * self.duration_weight) +
                          (aux_cif_loss * self.cif_weight))

            out = DualFusionOutput(
                loss = total_loss,
                logits = outputs.logits,
                loss_lm = outputs.loss.detach(),
                loss_ctc = aux_ctc_loss.detach(),
                loss_audio = norm_audio_loss.detach(),
                loss_duration = aux_duration_loss.detach()
            )

            return out

        finally:
            pass
            # for injection_layer in self.injection_layers:
            #     injection_layer.injection_audio = None
            #     injection_layer.injection_audio_mask = None
            #     injection_layer.prompt_audio = None
            #     injection_layer.prompt_audio_mask = None

    def generate(self, audios=None,
                 audio_masks=None,
                 audio_lb_tokens=None,
                 labels=None,
                 label_masks=None,
                 duration_ids=None,
                 index=None,
                 ctc_labels=None,
                 ctc_lengths=None,
                 tag_tokens=None,
                 tag_masks=None,
                 **kwargs):

        with torch.inference_mode():
            inputs = audios
            batch_size = inputs.shape[0]

            if isinstance(inputs, torch.Tensor):
                batch_features = inputs.to(self.device)
                batch_masks = audio_masks.to(self.device)

            elif isinstance(inputs, numpy.ndarray) or isinstance(inputs, list):
                batch_audio = self.processor(
                    inputs,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="max_length",
                    return_attention_mask=True,
                    truncation=True
                )

                batch_features = batch_audio.input_features.to(self.device)
                batch_masks = batch_audio.attention_mask.to(self.device)

            else:
                raise ValueError(f"Unexpected input type for generate: {type(inputs)}")

            encoder_outputs = self.speech_encoder(batch_features, attention_mask=batch_masks, output_hidden_states=True)

            audio_embeddings = encoder_outputs.last_hidden_state
            proj_embeddings = None
            down_embeddings = None
            down_masks = None

            if self.include_adapter:
                (proj_embeddings,
                 down_embeddings,
                 down_masks, _) = self.get_input_embeddings(audio_embeddings, batch_masks)

            prompt_embed, prompt_mask, _, _ = self._prepare_input_embeds(batch_size,
                                                                         proj_embeddings,
                                                                         down_masks,
                                                                         tag_tokens=tag_tokens,
                                                                         tag_masks=tag_masks)

            pad_token_id = self.language_tokenizer.pad_token_id
            input_ids = torch.ones(
                (prompt_embed.shape[0], prompt_embed.shape[1]),
                dtype=torch.long,
                device=self.device
            ) * pad_token_id

            injection_audios, injection_masks = self.inject(encoder_outputs,
                                                            audio_embeddings,
                                                            batch_masks,
                                                            down_embeddings,
                                                            down_masks)

            for injection_audio, injection_mask, injection_layer in zip(injection_audios, injection_masks, self.injection_layers):
                injection_layer.injection_audio = injection_audio
                injection_layer.injection_audio_mask = injection_mask
                injection_layer.prompt_audio = proj_embeddings
                injection_layer.prompt_audio_mask = down_masks

            try:
                outputs = self.language_model.generate(
                    input_ids=input_ids,
                    inputs_embeds=prompt_embed,
                    attention_mask=prompt_mask.bool(),
                    pad_token_id=pad_token_id,
                    **kwargs
                )

            finally:
                for injection_layer in self.injection_layers:
                    injection_layer.cross_attention_layer.clear_cache()
                    injection_layer.injection_audio = None
                    injection_layer.injection_audio_mask = None
                    injection_layer.prompt_audio = None
                    injection_layer.prompt_audio_mask = None

            return outputs

    @property
    def config(self):
        return self.language_model.config