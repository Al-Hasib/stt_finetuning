import os
import json
import requests
import yt_dlp
from pydub import AudioSegment
import logging
import pandas as pd
from datafetch import fetch_transcription
from concurrent.futures import ThreadPoolExecutor, as_completed


#================================ Ensure required directories exist ===============================
os.makedirs("jsonfiles", exist_ok=True)
os.makedirs("wavfiles", exist_ok=True)

# Setup logging for better debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class processAudio:

    # Function to check if a file exists
    def file_exists(self, filename):
        return os.path.exists(filename)


    #=============================== Function to download audio from video URL========================
    def download_audio(self, video_url, file_counter):
        wav_file = f'{file_counter}.wav'

        # Ensure audio download always proceeds for debugging
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'wavfiles/{file_counter}.%(ext)s',
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


    # Function to process each video URL
    def process_video(self, video_url, file_counter):
        wav_file = f'wavfiles/{file_counter}.wav'  # Use the file_counter-based name
        json_file = f'jsonfiles/{file_counter}_transcription.json'
        transcription = []  # Initialize transcription as an empty list

        # Download audio if not already downloaded
        if not self.file_exists(wav_file):
            self.download_audio(video_url, file_counter)  # Ensure the correct file name is passed

        # Check if the transcription file exists
        if not self.file_exists(json_file):
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
    def read_video_urls(self,file_path):
        with open(file_path, "r") as file:
            return file.readlines()

    # ================================ Process the URLs ================================
    def process_video_urls_concurrent(self, video_urls, saved_video_file, video_url_file, max_workers=4):
        """
        Process video URLs concurrently with a specified number of worker threads.
        """
        # Open the saved_video_url.txt to keep track of processed URLs
        with open(saved_video_file, "a") as saved_file:
            # Get the last file counter from the saved file, or start from 1
            last_saved_index = 0
            if os.path.exists(saved_video_file):
                with open(saved_video_file, "r") as saved_file_read:
                    lines = saved_file_read.readlines()
                    if lines:
                        last_saved_index = int(lines[-1].split('.')[0])  # Get the last index from saved file

            all_transcriptions = []
            futures = {}

            # Start a ThreadPoolExecutor with a specified number of workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks for processing videos
                for index, video_url in enumerate(video_urls, start=last_saved_index + 1):
                    video_url = video_url.strip()  # Clean the URL
                    if video_url:
                        logging.info(f"Submitting task for video URL: {video_url}")

                        # Append the URL to saved_video_url.txt with the corresponding index
                        saved_file.write(f"{index}. {video_url}\n")

                        # Submit a task to the executor
                        future = executor.submit(self.threaded_process_video, video_url, index)
                        futures[future] = index  # Map futures to their respective file counters

                # Process results as threads complete
                for future in as_completed(futures):
                    file_counter = futures[future]  # Get the file counter for the completed task
                    try:
                        _, transcription = future.result()  # Get the transcription from the future
                        all_transcriptions.extend(transcription)  # Collect all transcriptions
                        logging.info(f"Transcription for video {file_counter} processed.")
                    except Exception as e:
                        logging.error(f"Error retrieving result for video {file_counter}: {e}")

            # Clear the video_url.txt file after processing all URLs
            with open(video_url_file, "w") as file:
                file.write("")  # Clear the file after processing

            logging.info(f"All URLs have been saved to {saved_video_file} and removed from {video_url_file}.")

        return all_transcriptions


    def threaded_process_video(self, video_url, file_counter):
        """
        Process a single video URL and return the transcription.
        """
        try:
            transcription = self.process_video(video_url, file_counter)
            logging.info(f"Completed processing for video {file_counter}: {video_url}")
            return file_counter, transcription  # Return the index and transcription
        except Exception as e:
            logging.error(f"Error processing video {file_counter}: {e}")
            return file_counter, []



    # ============================ Chunks creation starts here =============================

    def create_audio_chunk(self,transcription_entry, audio, video_directory, chunk_index):
        """
        Create a single audio chunk from a transcription entry.
        """
        data = []
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
                return []

        # Append transcription and file path to the data list
        data.append({
            "text": text,
            "audio_file": audio_segment_filename  # This stores the path to the audio file
        })

        return data


    def process_chunks_concurrently(self, all_transcriptions, audio_file, video_index):
        """
        Process all audio chunks concurrently for a given audio file.
        """
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

        # Define a directory for the video
        video_directory = f"video_{video_index}"
        os.makedirs(video_directory, exist_ok=True)

        # Use ThreadPoolExecutor to process each transcription entry concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
            futures = {
                executor.submit(self.create_audio_chunk, entry, audio, video_directory, idx): idx
                for idx, entry in enumerate(all_transcriptions, start=1)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        data.extend(result)
                except Exception as e:
                    logging.error(f"Error processing chunk: {e}")

        return data