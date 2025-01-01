import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from libs.processaudio import AudioProcessor

def threaded_process_video(video_url, file_counter):
    """
    Process a single video URL and return the transcription.
    """
    try:
        transcription = AudioProcessor.process_video(video_url, file_counter)
        logging.info(f"Completed processing for video {file_counter}: {video_url}")
        return file_counter, transcription  # Return the index and transcription
    except Exception as e:
        logging.error(f"Error processing video {file_counter}: {e}")
        return file_counter, []
