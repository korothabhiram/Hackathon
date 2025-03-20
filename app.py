import queue
import numpy as np
import sounddevice as sd
import scipy.signal
from faster_whisper import WhisperModel
import tweepy

# Configuration parameters
CHUNK_DURATION = 5.0            # seconds of audio per transcription chunk
TARGET_SAMPLE_RATE = 16000      # Whisper expects 16 kHz audio
DEVICE_SAMPLE_RATE = 44100      # default microphone sample rate (adjust if needed)
CHANNELS = 1                    # mono audio

# Twitter API credentials (set as environment variables for security)
TWITTER_API_KEY = ("TWITTER_API_KEY")
TWITTER_API_SECRET = ("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = ("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = ("TWITTER_ACCESS_SECRET")

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
twitter_api = tweepy.API(auth)

# A thread-safe queue to hold recorded audio data
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback function that receives audio data from sounddevice."""
    if status:
        print(status)
    # If the input has more than one channel, convert to mono by averaging
    if indata.shape[1] > 1:
        indata = np.mean(indata, axis=1, keepdims=True)
    audio_queue.put(indata.copy())

# Initialize the faster-whisper model in CPU int8 mode.
CHUNK_DURATION = 1.0            # seconds of audio per transcription chunk
model = WhisperModel("base", device="cpu", compute_type="float32")
#The above values can be changed to tiny, small, largev2, largev3-turbo, Device can be cuda for nvidia GPUs, Compute type can be of int or float
def transcribe_audio(audio_chunk):
    """
    Resample the captured audio chunk (if needed) and run transcription.

    :param audio_chunk: numpy array with shape (n_samples, 1)
    """
    # Resample if the device sample rate does not match the target sample rate.
    if DEVICE_SAMPLE_RATE != TARGET_SAMPLE_RATE:
        num_samples = int(len(audio_chunk) * TARGET_SAMPLE_RATE / DEVICE_SAMPLE_RATE)
        # Flatten, resample, then convert back to float32.
        audio_chunk = scipy.signal.resample(audio_chunk.flatten(), num_samples).astype(np.float32)
    else:
        audio_chunk = audio_chunk.flatten().astype(np.float32)

    # Transcribe the audio chunk using faster-whisper.
    segments, info = model.transcribe(audio_chunk, beam_size=5)
    transcribed_text = " ".join([segment.text for segment in segments])
    print(transcribed_text)
    
    if transcribed_text.strip():
        tweet_text(transcribed_text)

def tweet_text(text):
    """Posts the transcribed text as a tweet."""
    try:
        twitter_api.update_status(text)
        print("Tweet posted successfully!")
    except tweepy.TweepError as e:
        print(f"Failed to post tweet: {e}")

def main():
    print("Starting live transcription. Press Ctrl+C to stop.")
    # Open an input stream using sounddevice.
    with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=DEVICE_SAMPLE_RATE):
        audio_buffer = np.empty((0, 1), dtype=np.float32)
        try:
            while True:
                # Accumulate audio data from the queue.
                while not audio_queue.empty():
                    data = audio_queue.get()
                    audio_buffer = np.vstack((audio_buffer, data))

                # When we have enough samples, process a chunk.
                if len(audio_buffer) >= CHUNK_DURATION * DEVICE_SAMPLE_RATE:
                    chunk_samples = int(CHUNK_DURATION * DEVICE_SAMPLE_RATE)
                    audio_chunk = audio_buffer[:chunk_samples]
                    # Remove the processed part from the buffer.
                    audio_buffer = audio_buffer[chunk_samples:]
                    # Transcribe the current audio chunk.
                    transcribe_audio(audio_chunk)
        except KeyboardInterrupt:
            print("\nTranscription stopped.")

if __name__ == "__main__":
    main()
