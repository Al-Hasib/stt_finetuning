from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
tokenizer = WhisperTokenizer.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000", language="Arabic", task="transcribe")
processor = WhisperProcessor.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000", language="Arabic", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000")



model.generation_config.language = "arabic"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

metric = evaluate.load("wer")


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v3-turbo_3",  # change to a repo name of your choice
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


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=atc_dataset_train,
    eval_dataset=atc_dataset_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
#     tokenizer=processor.feature_extractor,
)


trainer.train()

