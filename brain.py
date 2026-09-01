import ollama

# --- Step 2: System Prompt ---
# Defines Jarvis's personality and hard constraints.
SYSTEM_PROMPT = """You are Friday. 
Avoid giving lots of fluff or praise.
I want realistic answers.
I want short and concise answers.
If you do not know something or it does not exist, do not make anything up,
instead, simply state so shortly in a friendly manner.
You must answer in 2 sentences maximum."""

def construct_prompt(user_name, transcribed_text):
    """
    Step 3: Contextual Prompt Constructor
    Combines the detected user name with the transcribed speech.
    """
    return f"User {user_name} asks: {transcribed_text}"

def main():
    # Mock inputs matching your Phase 4 Acceptance Criteria
    mock_name = "Ethan"
    mock_command = "What do I have scheduled for today?"

    # Build the final prompt string
    contextual_prompt = construct_prompt(mock_name, mock_command)
    
    print("--- Pipeline Status ---")
    print("System Prompt Loaded.")
    print(f"Constructed Context: {contextual_prompt}")
    print("\nSending to Ollama... (Waiting for response)")

    try:
        # --- Step 1: Send to Ollama ---
        # Note: Change 'llama3' to 'phi3' if that is the model you pulled.
        response = ollama.chat(
            model='llama3', 
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': contextual_prompt}
            ]
        )

        # Extract and print the assistant's reply
        jarvis_reply = response['message']['content']
        print(f"\nJarvis: {jarvis_reply}\n")

    except Exception as e:
        print(f"\nError connecting to Ollama: {e}")
        print("Troubleshooting: Ensure the Ollama application is running in the background.")

if __name__ == "__main__":
    main()