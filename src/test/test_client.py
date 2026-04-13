
from src.llm_client import LLMClient
from src.vlm_client import VLMClient
from traceback import format_exc

class TestLLMClient(LLMClient):
    def __init__(self, **model_parameters):
        super().__init__(**model_parameters)

    def test_response(self):
        response = self(
            system_message="You are a helpful assistant that provides concise answers.",
            user_message="What is the capital of France?"
        )
        print("LLM Response:\n", response)
        self.get_total_usage()

class TestVLMClient(VLMClient):
    def __init__(self, **model_parameters):
        super().__init__(**model_parameters)

    def test_response_with_url_image(self):
        # image_url = "https://commons.wikimedia.org/wiki/File:Valentino_Rossi_2017.jpg#/media/File:Valentino_Rossi_2017.jpg"
        
        try:
            image_url = "https://share.google/vBs36JundlhvwOj6e"
            response = self(
                text_prompt="Who's the person in the image?",
                image=image_url
            )
            print("VLM Response:\n", response)
            self.get_total_usage()
        except Exception as e:
            print(f"Error during VLM test with URL image: {e}")
            print(format_exc())

    def test_response_with_local_image(self):

        try:
            image_path = '/home/agents/ProjectsWorkspace/FoundationClients/src/test/alex_delpiero.jpg'
            response = self(
                text_prompt="Who's the person in the image?",
                image=image_path
            )
            print("VLM Response:\n", response)
            self.get_total_usage()
        except Exception as e:
            print(f"Error during VLM test with local image: {e}")
            print(format_exc())