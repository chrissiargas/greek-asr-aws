import numpy as np
import wandb
import evaluate

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")
import random


def wrap_compute_metrics(tokenizer, dataset, writer, info):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        total_samples = len(predictions)

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        markdown_table = ''
        wandb_data = []

        max_safe_len = min(len(decoded_labels), len(dataset['eval']))
        num_samples = min(10, max_safe_len)
        sample_indices = random.sample(range(max_safe_len), num_samples)

        for i in sample_indices:
            raw_audio = dataset['eval'][i]['audio']['array']
            raw_audio = np.asarray(raw_audio, dtype=np.float32)
            audio_html = wandb.Audio(raw_audio, sample_rate=16000)

            reference = decoded_labels[i].replace('\n', ' ')
            prediction = decoded_preds[i].replace('\n', ' ')
            markdown_table += f' reference: {reference}\n prediction: {prediction}\n\n'
            wandb_data.append([audio_html, reference, prediction])

        if wandb.run is not None:
            writer.add_text('validation/sample_subset', markdown_table)
            table = wandb.Table(columns=['Audio', 'Reference', 'Prediction'], data=wandb_data)
            wandb.log({"validation/sample_predictions": table})

        metrics = {}
        if info['compute_wer_cer']:
            preds_arr = np.char.strip(np.array(decoded_preds))
            labels_arr = np.char.strip(np.array(decoded_labels))

            empty_ref_mask = (labels_arr == "")
            labels_arr[empty_ref_mask] = "<SIL>"

            preds_arr[empty_ref_mask] = np.where(
                preds_arr[empty_ref_mask] == "",
                "<SIL>",
                np.char.add("<SIL> ", preds_arr[empty_ref_mask])
            )

            preds_arr = np.where(preds_arr == "", " ", preds_arr)

            filtered_preds = preds_arr.tolist()
            filtered_labels = labels_arr.tolist()

            cer = cer_metric.compute(predictions=filtered_preds, references=filtered_labels)
            wer = wer_metric.compute(predictions=filtered_preds, references=filtered_labels)
            metrics = {"cer": cer, "wer": wer}

        return metrics

    return compute_metrics
