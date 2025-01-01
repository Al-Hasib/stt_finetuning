import os
import json
import requests
import yt_dlp
from pydub import AudioSegment
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from libs.datafetch import fetch_transcription

class AudioProcessor:
    def __init__(self):
        # Ensure required directories exist
        os.makedirs("libs/jsonfiles", exist_ok=True)
        os.makedirs("libs/wavfiles", exist_ok=True)
        os.makedirs("libs/csvFiles", exist_ok=True)

        # Setup logging for better debugging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    @staticmethod
    def file_exists(filename):
        return os.path.exists(filename)

    @staticmethod
    def download_audio(video_url, file_counter):
        wav_file = f'{file_counter}.wav'
        # Ensure audio download always proceeds for debugging
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'libs/wavfiles/{file_counter}.%(ext)s',
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

    def process_video(self, video_url, file_counter):
        wav_file = f'libs/wavfiles/{file_counter}.wav'
        json_file = f'libs/jsonfiles/{file_counter}_transcription.json'
        transcription = []

        if not self.file_exists(wav_file):
            self.download_audio(video_url, file_counter)

        if not self.file_exists(json_file):
            logging.warning(f"Transcription file {json_file} does not exist. Skipping transcription step.")
        else:
            with open(json_file, 'r', encoding='utf-8') as file:
                transcription = json.load(file)
                logging.info(f"{json_file} already exists. Skipping transcription download.")

        if not transcription:
            transcription = fetch_transcription(video_url, json_file, file_counter)

        return transcription

    @staticmethod
    def read_video_urls(file_path):
        with open(file_path, "r") as file:
            return file.readlines()

    @classmethod
    def process_video_urls_concurrent(cls, video_urls, saved_video_file, video_url_file, max_workers=4):
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
                        future = executor.submit(cls.threaded_process_video, video_url, index)
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
    
    @classmethod
    def threaded_process_video(cls, video_url, file_counter):
        try:
            transcription = cls().process_video(video_url, file_counter)  # Create an instance of the class
            logging.info(f"Completed processing for video {file_counter}: {video_url}")
            return file_counter, transcription  # Return the index and transcription
        except Exception as e:
            logging.error(f"Error processing video {file_counter}: {e}")
            return file_counter, []
   

    @staticmethod
    def create_audio_chunk(transcription_entry, audio, video_directory, chunk_index):
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

        # # Define a directory for the video
        # video_directory = f"video_{video_index}"
        # os.makedirs(video_directory, exist_ok=True)

        # counter=1

        #=================== Define a unique directory for the video============
        base_directory = f"libs/video_libs/video_{video_index}"
        video_directory = base_directory
        counter = 1
        while os.path.exists(video_directory):  # Ensure a unique directory name
            counter += 1
            video_directory = f"{base_directory}_{counter}"
        
        os.makedirs(video_directory, exist_ok=True)  # Create the unique directory

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