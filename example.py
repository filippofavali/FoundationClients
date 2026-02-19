import os
from llm_client import LLMClient
from vlm_client import VLMClient

# Mock API keys for demonstration if not present
if not os.getenv("GROQ_API_KEY"): os.environ["GROQ_API_KEY"] = "mock_groq_key"
if not os.getenv("OPENAI_API_KEY"): os.environ["OPENAI_API_KEY"] = "mock_openai_key"

def main():
    print("--- LLM Client Example ---")
    # Initialize Groq Client
    # Note: This will fail connection if key is invalid, so we wrap in try/except for demo purposed without real keys
    try:
        llm = LLMClient(model_name="groq/llama3-8b", temperature=0.7)
        print(f"Initialized LLM: {llm.model_name}")
        
        # In a real scenario, we would call:
        # response = llm("What is the capital of France?")
        # print("Response:", response)
        # llm.log_metrics()
        print("LLM Client initialized successfully.")
    except Exception as e:
        print(f"LLM Client init failed (expected if no valid key/sdk): {e}")

    print("\n--- VLM Client Example ---")
    try:
        vlm = VLMClient(model_name="openai/gpt-4o")
        print(f"Initialized VLM: {vlm.model_name}")
        
        # response = vlm("Describe this image", "https://example.com/image.jpg")
        # print("Response:", response)
        print("VLM Client initialized successfully.")
    except Exception as e:
         print(f"VLM Client init failed: {e}")

if __name__ == "__main__":
    main()
