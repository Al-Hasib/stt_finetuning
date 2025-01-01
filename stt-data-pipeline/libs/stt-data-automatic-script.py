import os
import json
import requests
import yt_dlp
from pydub import AudioSegment
import logging
import pandas as pd
from huggingface_hub import login
from huggingface_hub import whoami
import re
import unicodedata
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import Audio
from num2words import num2words
import time
import random
import string


# Setup logging for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check if a file exists
def file_exists(filename):
    return os.path.exists(filename)

# Function to download audio from video URL
def download_audio(video_url, file_counter):
    wav_file = f'{file_counter}.wav'

    # Ensure audio download always proceeds for debugging
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{file_counter}.%(ext)s',
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        logging.info(f"Audio {wav_file} downloaded.")
    except Exception as e:
        logging.error(f"Audio download failed for {wav_file}: {e}")
        return None

    return wav_file

# Function to fetch and parse transcription data from subtitles
def fetch_transcription(video_url, json_file, file_counter):
    transcription = []

    try:
        ydl_opts_info = {'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)

        # Get available subtitles (manual and automatic)
        subtitles = info_dict.get('subtitles', {})
        automatic_captions = info_dict.get('automatic_captions', {})

        subtitle_url = None

        # Check for manual Arabic subtitles
        if 'ar' in subtitles:
            subtitle_url = subtitles['ar'][0]['url']
            logging.info(f"Found manual Arabic subtitles: {subtitle_url}")
        elif 'ar' in automatic_captions:
            subtitle_url = automatic_captions['ar'][0]['url']
            logging.info(f"Found automatic Arabic subtitles: {subtitle_url}")

        if subtitle_url:
            # Fetch the subtitle data
            response = requests.get(subtitle_url)
            response.raise_for_status()

            subtitle_data = response.json()
            last_start_time = None

            # Parse subtitle events and generate transcription data
            for event in subtitle_data.get('events', []):
                start = event.get('tStartMs', 0) / 1000  # Convert to seconds
                text_segments = event.get('segs', [])
                text = ''.join(seg.get('utf8', '') for seg in text_segments)

                if text.strip():
                    if last_start_time is not None:
                        transcription[-1]["end"] = start

                    transcription.append({
                        "text": text.strip(),
                        "start": start,
                    })

                    last_start_time = start

            if transcription:
                transcription[-1]["end"] = last_start_time

            # Save the transcription to a JSON file
            with open(json_file, 'w', encoding='utf-8') as file:
                json.dump(transcription, file, ensure_ascii=False, indent=4)
            logging.info(f"Transcription saved to {json_file}")
        else:
            logging.warning("No Arabic subtitles found.")
    except Exception as e:
        logging.error(f"An error occurred during subtitle processing: {e}")

    return transcription

# Function to process each video URL
def process_video(video_url, file_counter):
    wav_file = f'{file_counter}.wav'  # Use the file_counter-based name
    json_file = f'{file_counter}_transcription.json'
    transcription = []  # Initialize transcription as an empty list

    # Download audio if not already downloaded
    if not file_exists(wav_file):
        download_audio(video_url, file_counter)  # Ensure the correct file name is passed

    # Check if the transcription file exists
    if not file_exists(json_file):
        logging.warning(f"Transcription file {json_file} does not exist. Skipping transcription step.")
    else:
        # If transcription file exists, load it
        with open(json_file, 'r', encoding='utf-8') as file:
            transcription = json.load(file)
            logging.info(f"{json_file} already exists. Skipping transcription download.")

    # If transcription file does not exist, fetch transcription data
    if not transcription:
        transcription = fetch_transcription(video_url, json_file, file_counter)

    return transcription

# ================================ Read the video URLs from video_url.txt =============================
def read_video_urls(file_path):
    with open(file_path, "r") as file:
        return file.readlines()

# ================================ Process the URLs ================================
def process_video_urls(video_urls, saved_video_file, video_url_file):
    # Open the saved_video_url.txt to keep track of processed URLs
    with open(saved_video_file, "a") as saved_file:
        # Get the last file counter from the saved file, or start from 1
        last_saved_index = 0
        if os.path.exists(saved_video_file):
            with open(saved_video_file, "r") as saved_file_read:
                lines = saved_file_read.readlines()
                if lines:
                    last_saved_index = int(lines[-1].split('.')[0])  # Get the last index from saved file

        # Process each URL and assign an incremental counter starting from the last used index
        all_transcriptions = []
        for index, video_url in enumerate(video_urls, start=last_saved_index + 1):
            video_url = video_url.strip()  # Clean the URL

            if video_url:
                logging.info(f"Processing video URL: {video_url}")

                # Append the URL to saved_video_url.txt with the corresponding index
                saved_file.write(f"{index}. {video_url}\n")

                # Process the video by downloading and saving audio and transcription
                transcription = process_video(video_url, index)
                all_transcriptions.extend(transcription)

        # Clear the video_url.txt file after processing all URLs
        with open(video_url_file, "w") as file:
            file.write("")  # Clear the file after processing

        logging.info(f"All URLs have been saved to {saved_video_file} and removed from {video_url_file}.")

    return all_transcriptions


# ============================ Chunks creation starts here =============================

def create_audio_chunks(all_transcriptions, audio_file, video_index):
    data = []

    # Check if the audio file exists before proceeding
    if not os.path.exists(audio_file):
        logging.error(f"Error: {audio_file} does not exist. Skipping chunk creation.")
        return []

    # Attempt to load the audio file once
    try:
        audio = AudioSegment.from_wav(audio_file)
    except Exception as e:
        logging.error(f"Error loading audio file {audio_file}: {e}")
        return []

    # Create a new directory for each .wav file using a simple counter starting from 1
    video_directory_counter = 1
    video_directory = f"{video_directory_counter}"

    # Ensure unique directory name for each .wav file
    while os.path.exists(video_directory):
        video_directory_counter += 1
        video_directory = f"{video_directory_counter}"

    os.makedirs(video_directory)

    # Iterate over the transcription entries to create audio chunks
    for chunk_index, transcription_entry in enumerate(all_transcriptions, start=1):
        text = transcription_entry['text']
        start_time = transcription_entry['start'] * 1000  # Convert to milliseconds
        end_time = transcription_entry['end'] * 1000  # Convert to milliseconds

        # Filename for the chunk
        audio_segment_filename = f"{video_directory}/{chunk_index}_{int(start_time)}_{int(end_time)}.wav"

        # Use the existing audio segment file if it exists
        if os.path.exists(audio_segment_filename):
            logging.info(f"Using existing audio chunk: {audio_segment_filename}")
        else:
            # Extract and export the audio segment only if the file doesn't exist
            try:
                audio_segment = audio[start_time:end_time]
                audio_segment.export(audio_segment_filename, format="wav")
                logging.info(f"Audio chunk created: {audio_segment_filename}")
            except Exception as e:
                logging.error(f"Error creating audio chunk {audio_segment_filename}: {e}")
                continue

        # Append transcription and file path to the data list
        data.append({
            "text": text,
            "audio_file": audio_segment_filename  # This stores the path to the audio file
        })

    return data


# ============================ Save final dataset to CSV ================================
def save_to_csv(data, base_stt_dataset_path):
    df = pd.DataFrame(data)

    # Check if the CSV file already exists and increment the counter for CSV file if necessary
    csv_counter = 1
    stt_dataset_path = base_stt_dataset_path
    while os.path.exists(stt_dataset_path):
        stt_dataset_path = f"stt_dataset_{csv_counter}.csv"
        csv_counter += 1

    df.to_csv(stt_dataset_path, index=False)

    # Print out where the files have been saved
    logging.info(f"Processed audio chunks and transcriptions saved to {stt_dataset_path}")


#=========================================== Function to log into Hugging Face================================================

def huggingFacelogin():
    login(token="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")
    print(whoami())

#========================= Function to normalize Arabic text (remove diacritics, unwanted symbols, and numbers)===============

def normalize_arabic(text):
    # Remove unwanted symbols (non-arabic characters)
    text = re.sub(r"[^\w\s\u0621-\u064A\u0660-\u0669]", "", text)

    # Normalize Arabic script (remove diacritics, etc.)
    text = unicodedata.normalize("NFKD", text)

    # Optional: Convert Arabic numbers to words
    text = re.sub(r"\d+", lambda x: num2words(x.group(), lang='ar'), text)

    return text.strip()


# ========================== Function to Generate Unique Dataset Name ===========================
def generate_unique_dataset_name(base_name="dataset"):
    # Generate a unique name using timestamp or random string
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Current timestamp
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))  # Random string
    
    # Combine base name with timestamp and random string
    unique_name = f"{base_name}_{timestamp}_{random_string}"
    return unique_name


# ========================== Push to Hugging Face ===========================
def push_to_huggingface(df, base_dataset_name="dataset"):
    # Generate a unique dataset name
    unique_dataset_name = generate_unique_dataset_name(base_dataset_name)
    
    # Apply normalization to the transcriptions
    df['text'] = df['text'].apply(normalize_arabic)
    
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

# Test Hugging Face push
# def push_data_to_hf(df):
#     # Specify the dataset name and directory
#     push_to_huggingface(df, "audio_chunks_directory", "arabic_speech_to_text")



# Main execution flow
def main():
    video_url_file = 'video_url.txt'
    saved_video_file = 'saved_video_url.txt'
    video_urls = read_video_urls(video_url_file)
    huggingFacelogin()

    if video_urls:
        all_transcriptions = process_video_urls(video_urls, saved_video_file, video_url_file)

        data = []
        # Iterate over the video URLs and corresponding transcriptions
        for video_index, video_url in enumerate(video_urls, start=1):  # Sequential directory names (1, 2, 3, ...)
            video_url = video_url.strip()  # Clean the URL
            audio_file = f"{video_index}.wav"  # Name audio files as "1.wav", "2.wav", ...

            # Define a unique directory for each video based on the counter
            video_directory = f"video_{video_index}"  # e.g., "video_1", "video_2", etc.

            # Check if the audio file exists before processing
            if not os.path.exists(audio_file):
                logging.error(f"Error: {audio_file} does not exist. Skipping chunk creation.")
                continue

            # Create audio chunks for the current video and save the results
            chunks_data = create_audio_chunks(all_transcriptions, audio_file, video_directory)
            data.extend(chunks_data)

        # Save the dataset to CSV
        save_to_csv(data, 'stt_dataset.csv')
        # Push the dataset to Hugging Face
        # Assuming `data` contains the audio chunks and transcriptions
        
        df = pd.DataFrame(data)
        
        # Push the dataset to Hugging Face
        push_to_huggingface(df,"arabic_speech_to_text")


if __name__ == "__main__":
    main()
