
from src.llm_client import LLMClient
from src.vlm_client import VLMClient
from traceback import format_exc
from PIL import Image
import json

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

    def test_response_with_local_image(self, image_path: str=None):

        try:
            image_path = image_path or '/home/agents/ProjectsWorkspace/FoundationClients/src/test/alex_delpiero.jpg'
            with Image.open(image_path) as image:
                pixels_width, pixels_height = image.size
                print(f"Image dimensions: {pixels_width} x {pixels_height}")

            bb_prompt = """
            Give bounding box coordinates for the hands in the image.
            The image is provided in the size of {pixels_width} x {pixels_height}.
            Strictly use the following json format for the response, avoid any additional text or explanation.

            {{
            "bounding_boxes": [
                {{
                    "label": "label-of-the-object-in-the-bounding-box",
                    "x_min": top-left-x-pixel,
                    "y_min": top-left-y-pixel,
                    "x_max": bottom-right-x-pixel,
                    "y_max": bottom-right-y-pixel
                }}, 
                ]
            }}
            """
            bb_prompt = bb_prompt.format(
                pixels_width=pixels_width, pixels_height=pixels_height
            )

            response = self(
                text_prompt=bb_prompt,
                image=image_path,
                force_json_response=True
            )
            print("VLM Response:\n", response)
            response_data = json.loads(response) if isinstance(response, str) else response
            self._draw_bbs(response_data.get("bounding_boxes", []), image_path, print=True)
            self.get_total_usage()
        except Exception as e:
            print(f"Error during VLM test with local image: {e}")
            print(format_exc())