from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import torch
from transformers import WhisperProcessor, AutoTokenizer

GREEK_CHARS = "αβγδεζηθικλμνξοπρστυφχψωςάέήίόύώϊϋΐΰ"
EN_CHARS = "abcdefghijklmnopqrstuvwxyz"
PUNCT = " '.,!?;:-"
NUMBERS = "0123456789"
ALL_CHARS = ["<pad>", "<unk>", "<blank>"] + list(GREEK_CHARS + EN_CHARS + PUNCT + NUMBERS)

CHAR2ID = {c: i for i, c in enumerate(ALL_CHARS)}
BLANK_IDX = CHAR2ID["<blank>"]
PAD_IDX = CHAR2ID["<pad>"]
UNK_IDX = CHAR2ID["<unk>"]
NUM_CLASSES = len(ALL_CHARS)

PROMPT_VERBATIM = "Κατά λέξη μεταγραφή: "
PROMPT_FORMAL = "Κανονικοποιημένη μεταγραφή: "

DATASET_PROMPT_MAP = {
    'common_voice': PROMPT_VERBATIM,
    'fleurs': PROMPT_VERBATIM,
    'logotypographia': PROMPT_VERBATIM,
    'stoma': PROMPT_VERBATIM
}

def compute_length(batch):
    return {"length": len(batch["audio"]["array"])}

@dataclass
class DataCollator(object):
    processor: WhisperProcessor
    language_tokenizer: AutoTokenizer
    device: str = None
    padding: bool = True
    truncation: bool = True
    has_audio_lb_tokens: bool = False
    has_duration_lb: bool = False
    duration_resolution: float = 0.1
    add_to_vocab: bool = False
    contain_index: bool = False
    to_chars: bool = False
    prompt_verbatim: bool = False

    def get_duration_id(self, duration):
        bin_idx = round(duration / self.duration_resolution)
        return bin_idx

    def get_transcript_with_duration_token(self, instance: Dict):
        bin_idx = self.get_duration_id(instance["duration"])
        duration_token = f"<|{bin_idx * self.duration_resolution:.2f}|>"
        transcript = f"{duration_token} {instance['reference']}" + self.language_tokenizer.eos_token
        return transcript

    def convert_to_chars(self, instance: Dict):
        text = instance['reference'].lower()
        char_ids = [CHAR2ID.get(c, UNK_IDX) for c in text]
        ctc_labels = torch.tensor(char_ids, dtype=torch.long)
        ctc_length = len(char_ids)

        return ctc_labels, ctc_length

    def __call__(self, instances: Sequence[Dict]):
        raw_audio = [x["audio"]["array"] for x in instances]

        batch_audio = self.processor(
            raw_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            return_attention_mask=True
        )

        if self.has_audio_lb_tokens:
            transcriptions = [x["reference"] + self.language_tokenizer.eos_token for x in instances]

            audio_lb_tokens = self.processor.tokenizer(
                transcriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=448,
                add_special_tokens=True
            )

        if self.has_duration_lb and self.add_to_vocab:
            transcriptions = [self.get_transcript_with_duration_token(x) for x in instances]
        else:
            transcriptions = [x["reference"] + self.language_tokenizer.eos_token for x in instances]

        if self.prompt_verbatim:
            tags = list(map(lambda x: DATASET_PROMPT_MAP.get(x.get('dataset_name'), PROMPT_FORMAL), instances))

            tag_tokens = self.language_tokenizer(
                tags,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            )

        # noinspection PyCallingNonCallable
        label_tokens = self.language_tokenizer(
            transcriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=False
        )
        if self.contain_index:
            indices = [x["index"] for x in instances]

        if self.to_chars:
            CTC_labels, CTC_lengths = [], []
            for instance in instances:
                ctc_labels, ctc_length = self.convert_to_chars(instance)
                CTC_labels.append(ctc_labels)
                CTC_lengths.append(ctc_length)

            CTC_labels_padded = torch.nn.utils.rnn.pad_sequence(CTC_labels, batch_first=True, padding_value=PAD_IDX)
            CTC_lengths = torch.tensor(CTC_lengths, dtype=torch.long)


        if self.device is not None:
            xy = {
                "audios": batch_audio.input_features.to(self.device),
                "audio_masks": batch_audio.attention_mask.to(self.device),
                "labels": label_tokens.input_ids.to(self.device),
                "label_masks": label_tokens.attention_mask.to(self.device)
            }

            if self.has_audio_lb_tokens:
                xy["audio_lb_tokens"] = audio_lb_tokens.input_ids.to(self.device)

            if self.has_duration_lb and not self.add_to_vocab:
                duration_ids = torch.Tensor([self.get_duration_id(x['duration']) for x in instances])
                xy["duration_ids"] = duration_ids.to(self.device)

            if self.contain_index:
                xy["index"] = torch.Tensor(indices).to(self.device)

            if self.to_chars:
                xy['ctc_labels'] = CTC_labels_padded.to(self.device)
                xy['ctc_lengths'] = CTC_lengths.to(self.device)

            if self.prompt_verbatim:
                xy['tag_tokens'] = tag_tokens.input_ids
                xy['tag_masks'] = tag_tokens.attention_mask

        else:
            xy = {
                "audios": batch_audio.input_features,
                "audio_masks": batch_audio.attention_mask,
                "labels": label_tokens.input_ids,
                "label_masks": label_tokens.attention_mask
            }

            if self.has_audio_lb_tokens:
                xy["audio_lb_tokens"] = audio_lb_tokens.input_ids

            if self.has_duration_lb and not self.add_to_vocab:
                duration_ids = torch.Tensor([self.get_duration_id(x['duration']) for x in instances])
                xy["duration_ids"] = duration_ids

            if self.contain_index:
                xy["index"] = torch.Tensor(indices)

            if self.to_chars:
                xy['ctc_labels'] = CTC_labels_padded
                xy['ctc_lengths'] = CTC_lengths

            if self.prompt_verbatim:
                xy['tag_tokens'] = tag_tokens.input_ids
                xy['tag_masks'] = tag_tokens.attention_mask

        return xy


