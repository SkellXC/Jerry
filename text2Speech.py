import pyttsx3
import re

def setup_voice():
    """Initializes the TTS engine and configures the voice."""
    engine = pyttsx3.init()
    # Adjust speaking rate (default is ~200; 175 sounds slightly more natural)
    engine.setProperty('rate', 175)
    return engine

def speak_full_text(engine, text):
    """Step 1: Speaks an entire block of text at once."""
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

def stream_sentences(engine, text_block):
    """
    Step 2: Simulates streaming by splitting a paragraph into sentences.
    In Phase 6, you will feed the LLM's live string stream into a function like this.
    """
    # Split text by punctuation (. ! ?) followed by a space
    sentences = re.split(r'(?<=[.!?]) +', text_block)
    
    for sentence in sentences:
        if sentence.strip():
            print(f"Streaming chunk: {sentence}")
            engine.say(sentence)
            engine.runAndWait()

def main():
    jarvis_voice = setup_voice()
    
    print("--- Testing Step 1: Basic TTS ---")
    basic_string = "Hello sir. I am currently offline and running locally."
    speak_full_text(jarvis_voice, basic_string)
    
    print("\n--- Testing Step 2: Sentence Streaming ---")
    mock_llm_generation = "I am processing your request now. The weather appears to be clear. All local systems are operating at peak efficiency."
    stream_sentences(jarvis_voice, mock_llm_generation)

if __name__ == "__main__":
    main()