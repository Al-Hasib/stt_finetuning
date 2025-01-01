import logging
import yt_dlp
import requests
import json
import pandas as pd
import os

os.makedirs("libs/csvFiles", exist_ok=True)

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


# ============================ Save final dataset to CSV ================================
def save_to_csv(data, base_stt_dataset_path="libs/csvFiles/stt_dataset.csv"):
    df = pd.DataFrame(data)

    # Check if the CSV file already exists and increment the counter for CSV file if necessary
    csv_counter = 1
    stt_dataset_path = base_stt_dataset_path
    while os.path.exists(stt_dataset_path):
        stt_dataset_path = f"libs/csvFiles/stt_dataset_{csv_counter}.csv"
        csv_counter += 1

    df.to_csv(stt_dataset_path, index=False)

    # Print out where the files have been saved
    logging.info(f"Processed audio chunks and transcriptions saved to {stt_dataset_path}")
