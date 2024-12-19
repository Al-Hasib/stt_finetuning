from transformers import (WhisperFeatureExtractor, 
                          WhisperProcessor,
                          WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          WhisperTokenizer)

import evaluate
import os
from src.data_processing import processing
import argparse
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from evaluate import load
from datasets import load_dataset, concatenate_datasets
from src.extract_dataset import ExtractNewDataset


def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features)[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# def parse_input():
#     parser = argparse.ArgumentParser(description="input as a string or a list of strings.")
#     # Use nargs='+' to accept one or more arguments (as a list)
#     parser.add_argument('input', type=str, nargs='+', help="name of datasets")
#     # Parse the arguments
#     args = parser.parse_args()
#     # If only one value is provided, it will be a string; if multiple values, it will be a list
#     return args.input



def create_dynamic_folder(base_path):
    # Get the list of existing folders
    existing_folders = os.listdir(base_path)
    
    # Filter for folders that follow the naming pattern "folder_<number>"
    folder_numbers = []
    for folder in existing_folders:
        if folder.startswith("model_") and folder[6:].isdigit():
            folder_numbers.append(int(folder[6:]))  # Extract the number

    # Determine the next folder number
    next_number = max(folder_numbers) + 1 if folder_numbers else 1
    new_folder_name = f"model_{next_number}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    
    # Create the new folder
    os.makedirs(new_folder_path)
    print(f"Created folder: {new_folder_name}")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Example usage
if __name__ == "__main__":
    # user_input = parse_input()
    extract_dataset = ExtractNewDataset()
    user_input = extract_dataset.get_new_items()
    print(user_input)
    
    # Check if the input is a list or a simple string
    if len(user_input) == 1:
        stt_data = user_input[0]
    else:
        stt_data = user_input
    
    print(stt_data)
    base_path = "./model_checkpoint"  # Replace with your folder path
    os.makedirs(base_path, exist_ok=True) 
    path_model = create_dynamic_folder(base_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    tokenizer = WhisperTokenizer.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000", language="Arabic", task="transcribe")
    processor = WhisperProcessor.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000", language="Arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000")
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    metric = evaluate.load("wer")
    print("parameter loading done")

    data_processing = processing(stt_data, feature_extractor,tokenizer)
    train_dataset, valid_dataset, test_dataset = data_processing.load_data()
    print('dataset loading successful')

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{path_model}",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=8000,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
    #     report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # cron job
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    #     tokenizer=processor.feature_extractor,
    )


    trainer.train()
    print("Training finished \n\n\n")

    trained_model = WhisperForConditionalGeneration.from_pretrained(path_model)

    if isinstance(stt_data, list):
        all_ds = []
        for stt_data_file in stt_data:
            ds = load_dataset(stt_data_file, split="test")
            all_ds.append(ds)
        final_ds = concatenate_datasets(all_ds)
        print(final_ds)

        result = final_ds.map(map_to_pred)
    
    else:
        final_ds = load_dataset(stt_data, split="test")
        result = final_ds.map(map_to_pred)

    wer = load("wer")
    wer_result = (100 * wer.compute(references=result["reference"], predictions=result["prediction"]))

    print(f"Word Error Rate: {wer_result}")

    

