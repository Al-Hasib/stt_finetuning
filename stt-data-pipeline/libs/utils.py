import os
import logging
import json
import requests
import shutil
from pathlib import Path

# Function to configure logging
def configure_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", mode='a')
        ]
    )
    logging.info("Logging is configured.")

# Function to read video URLs from a file
def read_video_urls(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return []
    
    with open(file_path, "r") as file:
        video_urls = file.readlines()
        logging.info(f"Read {len(video_urls)} video URLs from {file_path}")
        return video_urls

# Function to check if a file exists
def file_exists(file_path):
    return os.path.exists(file_path)

# Function to ensure a directory exists (creates it if not)
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Directory {directory_path} created.")
    else:
        logging.info(f"Directory {directory_path} already exists.")

# Function to download a file (e.g., audio, video)
def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses (e.g., 404)
        
        with open(save_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)
        
        logging.info(f"Downloaded file from {url} to {save_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading file: {e}")

# Function to read JSON data from a file
def read_json_file(file_path):
    if not file_exists(file_path):
        logging.error(f"JSON file {file_path} not found.")
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            logging.info(f"Successfully read JSON data from {file_path}")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}")
            return None

# Function to write data to a JSON file
def write_json_file(file_path, data):
    ensure_directory_exists(os.path.dirname(file_path))  # Ensure the directory exists
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        logging.info(f"Successfully wrote data to {file_path}")
    except IOError as e:
        logging.error(f"Error writing to JSON file {file_path}: {e}")

# Function to append data to an existing CSV file
def append_to_csv(file_path, data, fieldnames=None):
    ensure_directory_exists(os.path.dirname(file_path))  # Ensure the directory exists
    import csv
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if os.path.getsize(file_path) == 0:  # Write header if the file is empty
                writer.writeheader()
            writer.writerows(data)
        logging.info(f"Appended data to {file_path}")
    except IOError as e:
        logging.error(f"Error appending to CSV file {file_path}: {e}")

# Function to delete a file
def delete_file(file_path):
    if file_exists(file_path):
        os.remove(file_path)
        logging.info(f"Deleted file {file_path}")
    else:
        logging.warning(f"File {file_path} does not exist, skipping delete.")

# Function to fetch transcription or some other remote data
def fetch_transcription(url, json_file, file_counter):
    # Placeholder for actual fetching logic (like an API call)
    logging.info(f"Fetching transcription for {url}")
    transcription = {"transcription": f"Transcription for video {file_counter}"}
    write_json_file(json_file, transcription)  # Save to file after fetching
    return transcription

# Function to process video (downloading audio, fetching transcription)
def process_video(video_url, file_counter):
    wav_file = f'{file_counter}.wav'
    json_file = f'{file_counter}_transcription.json'
    transcription = []

    if not file_exists(wav_file):
        logging.info(f"Downloading audio for video {file_counter}")
        download_file(video_url, wav_file)

    if not file_exists(json_file):
        logging.warning(f"Transcription file {json_file} does not exist. Skipping transcription step.")
    else:
        transcription = read_json_file(json_file)
        if transcription:
            logging.info(f"Transcription file {json_file} found. Skipping transcription download.")
        else:
            transcription = fetch_transcription(video_url, json_file, file_counter)

    return transcription

# Additional utility functions can be added here...
