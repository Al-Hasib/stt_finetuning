
from huggingface_hub import login
from huggingface_hub import whoami
from datasets import Dataset, DatasetDict
from datasets import Audio
from num2words import num2words
from dotenv import load_dotenv
import unicodedata
import re
import time
import random
import string
import os
import logging

#=========================================== Function to log into Hugging Face================================================

# def huggingFacelogin():
#     login(token="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")
#     print(whoami())


class HuggingFaceUploader:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.token = os.getenv("HF_TOKEN")
        self.check_interval = os.getenv("CHECK_INTERVAL")
        
        if self.token:
            login(token=self.token)
            print(whoami())
        else:
            print("Hugging Face token not found in environment variables.")
        
        print(f"Check interval is set to: {self.check_interval}")

    def huggingFacelogin(self):
        if self.token:
            login(token=self.token)
            print(whoami())
        else:
            print("Hugging Face token not found in environment variables.")

    def normalize_arabic(self, text):
        # Remove unwanted symbols (non-arabic characters)
        text = re.sub(r"[^\w\s\u0621-\u064A\u0660-\u0669]", "", text)

        # Normalize Arabic script (remove diacritics, etc.)
        text = unicodedata.normalize("NFKD", text)

        # Optional: Convert Arabic numbers to words
        text = re.sub(r"\d+", lambda x: num2words(x.group(), lang='ar'), text)

        return text.strip()

    def generate_unique_dataset_name(self, base_name="dataset"):
        # Generate a unique name using timestamp or random string
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp
        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))  # Random string

        # Combine base name with timestamp and random string
        unique_name = f"{base_name}_{timestamp}_{random_string}"
        return unique_name

    def push_to_huggingface(self, df, base_dataset_name="dataset"):
        # Generate a unique dataset name
        unique_dataset_name = self.generate_unique_dataset_name(base_dataset_name)

        # Apply normalization to the transcriptions
        df['text'] = df['text'].apply(self.normalize_arabic)

        # Initialize a list to store the processed data
        data = []

        # Iterate over the rows of the dataframe
        for idx, row in df.iterrows():
            # Construct the correct audio path (assuming audio files are in the current directory or relative path)
            audio_filename = row["audio_file"].strip()
            audio_path = audio_filename  # Assuming audio files are in the same directory or relative path

            # Check if the audio file exists in the specified directory (current directory or relative path)
            if os.path.exists(audio_path):
                data.append({
                    "audio": audio_path,
                    "text": row["text"]
                })
            else:
                logging.warning(f"Audio file '{audio_filename}' not found at path '{audio_path}'")

        # Check if data is populated
        if len(data) == 0:
            logging.error("No valid data found. Please check the audio paths.")
            return

        # Create a Hugging Face Dataset from the list of dictionaries
        dataset = Dataset.from_list(data)

        # Load audio feature (this will automatically handle the .wav files)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Split the dataset
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        test_val_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

        # Combine the splits back into a DatasetDict
        dataset_dict = DatasetDict({
            "train": train_test_split["train"],
            "test": test_val_split["test"],
            "validation": test_val_split["train"]
        })

        # Push the dataset to Hugging Face Hub
        try:
            dataset_dict.push_to_hub(unique_dataset_name)
            logging.info(f"Dataset successfully pushed to Hugging Face Hub: {unique_dataset_name}")
        except Exception as e:
            logging.error(f"Error while pushing dataset: {e}")
            return
