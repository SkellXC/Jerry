import io
import wave
import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig

def setup_voice(model_path):
    print(f"Loading custom voice from {model_path}")
    return PiperVoice.load(model_path)  # auto-loads model_path + ".json"

def stream_speech(voice, text, syn_config):
    print(f"\nJarvis: {text}")
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    audio_buffer.seek(0)
    with wave.open(audio_buffer, 'rb') as wav_file:
        raw_data = wav_file.readframes(wav_file.getnframes())
        sample_rate = wav_file.getframerate()

    if not raw_data:
        raise RuntimeError(f"No audio was produced for: {text!r}")

    audio_array = np.frombuffer(raw_data, dtype=np.int16)
    sd.play(audio_array, samplerate=sample_rate)
    sd.wait()

def main():
    MODEL_PATH2 = r"C:\Users\ethan\Downloads\jarvis.onnx"
    try:
        jarvis_voice = setup_voice(MODEL_PATH2)
        syn_config = SynthesisConfig(length_scale=1.1)

        print("\n--- Testing Phase 5: RAM Streaming ---")
        mock_sentences = [
            "I am online and ready.",
            "My custom neural voice is fully operational.",
            "Notice how there is virtually zero latency before I begin speaking.",
            "A rainbow is a meteorological phenomenon that is caused by reflection, "
            "refraction and dispersion of light in water droplets resulting in a spectrum"
            " of light appearing in the sky."
        ]
        for sentence in mock_sentences:
            stream_speech(jarvis_voice, sentence, syn_config)

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()