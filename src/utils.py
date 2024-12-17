import evaluate
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
metric = evaluate.load("wer")


