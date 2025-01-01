import logging
import os
import pandas as pd
import time
from dotenv import load_dotenv
from libs.processaudio import AudioProcessor
from libs.huggingfacecred import HuggingFaceUploader
from libs.datafetch import fetch_transcription,save_to_csv
from libs.threadpool import threaded_process_video
from libs.utils import (
    configure_logging,        # Logging configuration
    read_video_urls,          # Reading video URLs
    file_exists,              # Checking if a file exists
    ensure_directory_exists,  # Ensuring directories exist
    download_file,            # File download utility
    read_json_file,           # Read JSON file
    write_json_file,          # Write to JSON file
    append_to_csv,            # Append to CSV file
    delete_file,              # Delete a file
    fetch_transcription,      # Fetch transcription data
    process_video             # Process video (download and transcription)
)

#===================== Load environment variables=====================
load_dotenv()

print("main.py is running...")

#===================== Configure logging==============================
configure_logging()


#==================== constants=======================================
WAV_FOLDER = 'libs/wavfiles'
JSON_FOLDER = 'libs/jsonfiles'
CSV_FOLDER = 'libs/csvFiles'
VIDEO_URL_FILE = 'video_url.txt'
SAVED_VIDEO_FILE = 'saved_video_url.txt'
CSV_FILE_PATH = os.path.join(CSV_FOLDER, 'stt_dataset.csv')
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 300)) 


# Main execution flow
def main():
    #======================== Ensure necessary directories exist=====================
    ensure_directory_exists(WAV_FOLDER)
    ensure_directory_exists(JSON_FOLDER)
    ensure_directory_exists(CSV_FOLDER)

    # Create an instance of AudioProcessor
    processor = AudioProcessor()

    # Instantiate the class
    uploader = HuggingFaceUploader()

    uploader.huggingFacelogin()

    while True:
        logging.info("Checking for new video URLs...")

        # Read video URLs
        video_urls = read_video_urls(VIDEO_URL_FILE)

        if video_urls:
            all_transcriptions = AudioProcessor.process_video_urls_concurrent(video_urls, SAVED_VIDEO_FILE, VIDEO_URL_FILE)

            data = []
            for video_index, video_url in enumerate(video_urls, start=1):
                video_url = video_url.strip()
                audio_file = os.path.join(WAV_FOLDER, f"{video_index}.wav")
                video_directory = os.path.join(WAV_FOLDER, f"video_{video_index}")

                if not os.path.exists(audio_file):
                    logging.error(f"Error: {audio_file} does not exist. Skipping chunk creation.")
                    continue

                # Call process_chunks_concurrently on the instance (processor)
                chunks_data = processor.process_chunks_concurrently(all_transcriptions, audio_file, video_index)
                data.extend(chunks_data)

            # Save the dataset to CSV
            save_to_csv(data, CSV_FILE_PATH)

            # Push the dataset to Hugging Face
            df = pd.DataFrame(data)
            uploader.push_to_huggingface(df, "arabic_speech_to_text")

        else:
            logging.info("No new video URLs found.")

        logging.info(f"Sleeping for {CHECK_INTERVAL // 60} minutes before the next check...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
