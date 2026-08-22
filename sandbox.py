import os
import struct
import wave
import math
import pyaudio
import numpy as np
from openwakeword.model import Model
from faster_whisper import WhisperModel

# --- Configuration ---
WAKE_WORD_MODEL = r"C:\Users\ethan\Jerry\Jerry\Hey_Friday_20260713_201149.onnx"
WAKE_WORD_MODEL2 = r"C:\Users\ethan\Downloads\Hey_Friday_20260713_201149.onnx"

# Audio Recording Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Required by OpenWakeWord and Whisper
CHUNK = 1280  # OpenWakeWord processes audio in 80ms chunks (1280 frames)
SILENCE_THRESHOLD = 500  # Adjust this based on your room's background noise
SILENCE_DURATION = 1.5   # Seconds of silence required to stop recording

# --- Initialize Local Models ---
print("Loading Faster-Whisper model...")
# Your Ryzen 7 5800X handles the 'base' model on CPU instantly
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

print("Loading OpenWakeWord model...")
# Downloads/loads the pre-trained 'Hey Jarvis' model locally
oww_model = Model(wakeword_models=[WAKE_WORD_MODEL2],
                   inference_framework="onnx")

def is_silent(data_chunk):
    """Calculates RMS (Root Mean Square) volume to detect silence."""
    count = len(data_chunk) / 2
    format_string = f"<{int(count)}h"
    shorts = struct.unpack(format_string, data_chunk)
    sum_squares = sum(s**2 for s in shorts)
    rms = math.sqrt(sum_squares / count)
    return rms < SILENCE_THRESHOLD

def record_command(audio_stream):
    """Records audio until silence is detected."""
    print("\nListening for command...")
    frames = []
    silent_chunks = 0
    max_silent_chunks = int((RATE / CHUNK) * SILENCE_DURATION)
    has_spoken = False

    while True:
        data = audio_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        if is_silent(data):
            if has_spoken:
                silent_chunks += 1
            if silent_chunks > max_silent_chunks:
                print("Silence detected. Stopping recording.")
                break
        else:
            has_spoken = True
            silent_chunks = 0  # Reset silence counter when speech is detected

    # Save the buffer to a temporary WAV file for Whisper
    filename = "temp_command.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    return filename

def main():
    # Set up PyAudio input stream
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("\nSystem ready. Say 'Hey Jarvis' to wake me up. Press Ctrl+C to exit.")

    try:
        while True:
            # Read exactly the number of frames OpenWakeWord expects
            pcm = audio_stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert raw audio bytes to numpy array for OpenWakeWord
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            
            # Process the audio frame
            prediction = oww_model.predict(audio_data)
            
            # OpenWakeWord returns a dictionary of scores. We grab the highest score.
            # 0.5 is the default confidence threshold for a positive detection.
            max_score = max(prediction.values())
            
            if max_score > 0.5:
                print("\nWake word detected!")
                
                # Step 2: Trigger recording loop
                audio_file = record_command(audio_stream)
                
                # Step 3: Transcribe with Faster-Whisper
                print("Transcribing...")
                segments, info = whisper_model.transcribe(audio_file, beam_size=5)
                
                transcription = "".join([segment.text for segment in segments]).strip()
                print(f"User command: {transcription}")
                
                # Clear the wake word model's internal audio buffer so it doesn't immediately re-trigger
                oww_model.reset()
                
                # Reset for the next command
                print("\nSystem ready. Say 'Hey Jarvis' to wake me up.")

    except KeyboardInterrupt:
        print("\nShutting down Hearing Module...")
    finally:
        # Clean up resources
        if 'audio_stream' in locals():
            audio_stream.stop_stream()
            audio_stream.close()
        if 'pa' in locals():
            pa.terminate()
        if os.path.exists("temp_command.wav"):
            os.remove("temp_command.wav")

if __name__ == "__main__":
    main()