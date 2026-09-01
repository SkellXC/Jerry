import wave
from piper.voice import PiperVoice

def main():
    # --- Configuration ---
    MODEL_PATH = r"C:\Users\ethan\Jerry\Jerry\jarvis.onnx"
    OUTPUT_FILE = r"C:\Users\ethan\Jerry\Jerry\jarvis_test.wav"
    
    print(f"Loading custom voice from {MODEL_PATH}...")
    try:
        jarvis_voice = PiperVoice.load(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    text = "I am online and ready. All systems are operational."
    
    print(f"\nJarvis: {text}")
    print("Synthesizing audio directly to your hard drive...")
    
    try:
        # We bypass RAM completely and write straight to a physical file
        with wave.open(OUTPUT_FILE, 'wb') as wav_file:
            jarvis_voice.synthesize(text, wav_file)
            
        print(f"\nSUCCESS! I have created a physical audio file at:")
        print(f"-> {OUTPUT_FILE}")
        print("Please open your folder, double-click this file, and see if Windows can play it.")
        
    except Exception as e:
        print(f"\n[CRITICAL PIPER ERROR]: {e}")
        print("If you see this, Piper crashed while trying to write the file.")

if __name__ == "__main__":
    main()