import logging
import yt_dlp
import requests
import json
import pandas as pd
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from stt_data.processing import processAudio



class utils:

    # Function to read video URLs from a file
    def read_video_urls(self, file_path):
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return []
        
        with open(file_path, "r") as file:
            video_urls = file.readlines()
            logging.info(f"Read {len(video_urls)} video URLs from {file_path}")
            return video_urls
        
    # Function to check if a file exists
    def file_exists(self, file_path):
        return os.path.exists(file_path)

    # Function to ensure a directory exists (creates it if not)
    def ensure_directory_exists(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Directory {directory_path} created.")
        else:
            logging.info(f"Directory {directory_path} already exists.")

    # Function to download a file (e.g., audio, video)
    def download_file(self, url, save_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses (e.g., 404)
            
            with open(save_path, "wb") as file:
                shutil.copyfileobj(response.raw, file)
            
            logging.info(f"Downloaded file from {url} to {save_path}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading file: {e}")

    # Function to read JSON data from a file
    def read_json_file(self, file_path):
        if not self.file_exists(file_path):
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
    def write_json_file(self, file_path, data):
        self.ensure_directory_exists(os.path.dirname(file_path))  # Ensure the directory exists
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            logging.info(f"Successfully wrote data to {file_path}")
        except IOError as e:
            logging.error(f"Error writing to JSON file {file_path}: {e}")

    # Function to append data to an existing CSV file
    def append_to_csv(self,file_path, data, fieldnames=None):
        self.ensure_directory_exists(os.path.dirname(file_path))  # Ensure the directory exists
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
    def delete_file(self,file_path):
        if self.file_exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted file {file_path}")
        else:
            logging.warning(f"File {file_path} does not exist, skipping delete.")

    # Function to fetch transcription or some other remote data
    def fetch_transcription(self,url, json_file, file_counter):
        # Placeholder for actual fetching logic (like an API call)
        logging.info(f"Fetching transcription for {url}")
        transcription = {"transcription": f"Transcription for video {file_counter}"}
        self.write_json_file(json_file, transcription)  # Save to file after fetching
        return transcription

    # Function to process video (downloading audio, fetching transcription)
    def process_video(self, video_url, file_counter):
        wav_file = f'{file_counter}.wav'
        json_file = f'{file_counter}_transcription.json'
        transcription = []

        if not self.file_exists(wav_file):
            logging.info(f"Downloading audio for video {file_counter}")
            self.download_file(video_url, wav_file)

        if not self.file_exists(json_file):
            logging.warning(f"Transcription file {json_file} does not exist. Skipping transcription step.")
        else:
            transcription = self.read_json_file(json_file)
            if transcription:
                logging.info(f"Transcription file {json_file} found. Skipping transcription download.")
            else:
                transcription = self.fetch_transcription(video_url, json_file, file_counter)

        return transcription
    

    def threaded_process_video(video_url, file_counter):
        """
        Process a single video URL and return the transcription.
        """
        try:
            transcription = processAudio.process_video(video_url, file_counter)
            logging.info(f"Completed processing for video {file_counter}: {video_url}")
            return file_counter, transcription  # Return the index and transcription
        except Exception as e:
            logging.error(f"Error processing video {file_counter}: {e}")
            return file_counter, []

   