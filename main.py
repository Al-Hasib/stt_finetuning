import logging
import pandas as pd
from src.stt_data.processing import processAudio
from src.stt_data.hf_credential import Credential
from src.stt_data.utils import utils
from src.stt_data.fetch import fetchData
import os


from utils import (
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



configure_logging()


# Main execution flow
def main():

    wav_folder = 'wavfiles'
    json_folder = 'jsonFiles'
    csv_folder = 'csvFiles'

    video_url_file = 'video_url.txt'
    saved_video_file = 'saved_video_url.txt'
    video_urls = read_video_urls(video_url_file)
    huggingFacelogin()

    if video_urls:
        all_transcriptions = process_video_urls_concurrent(video_urls, saved_video_file, video_url_file)

        data = []
        # Iterate over the video URLs and corresponding transcriptions
        for video_index, video_url in enumerate(video_urls, start=1):  # Sequential directory names (1, 2, 3, ...)
            video_url = video_url.strip()  # Clean the URL


            # audio_file = f"{video_index}.wav"  # Name audio files as "1.wav", "2.wav", ...

            # # Define a unique directory for each video based on the counter
            # video_directory = f"video_{video_index}"  # e.g., "video_1", "video_2", etc.

            # Define paths for the audio file and the video directory

            audio_file = os.path.join(wav_folder, f"{video_index}.wav")
            video_directory = os.path.join(wav_folder, f"video_{video_index}")



            # Check if the audio file exists before processing
            if not os.path.exists(audio_file):
                logging.error(f"Error: {audio_file} does not exist. Skipping chunk creation.")
                continue

            # Create audio chunks for the current video and save the results
            chunks_data = process_chunks_concurrently(all_transcriptions, audio_file, video_directory)
            data.extend(chunks_data)

        # Save the dataset to CSV
        # save_to_csv(data, 'stt_dataset.csv')
        # Push the dataset to Hugging Face
        # Assuming `data` contains the audio chunks and transcriptions

        base_csv_path = os.path.join(csv_folder, 'stt_dataset.csv')
        save_to_csv(data, base_csv_path)
        
        df = pd.DataFrame(data)
        
        # Push the dataset to Hugging Face
        push_to_huggingface(df,"arabic_speech_to_text")


if __name__ == "__main__":
    main()
