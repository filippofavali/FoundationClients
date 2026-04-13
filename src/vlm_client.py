import os, sys, base64, requests, base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io import BytesIO
from typing import Union
from PIL import Image
try:    
    from google import genai
except ImportError:
    genai = None
from base_client import BaseFoundationClient

class VLMClient(BaseFoundationClient):
    """
    Client for Vision-Language tasks.
    """

    def __init__(self, **model_parameters):
        super().__init__(**model_parameters)
    
    def _encode_image(self, image_source: Union[str, bytes, Image.Image]) -> str:
        """Encodes image to base64 string."""
        if isinstance(image_source, Image.Image):
            buffered = BytesIO()
            image_source.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image_source, str):
            if image_source.startswith("http"):
                return image_source # Return URL directly if provider supports it, or download and encode
            elif os.path.isfile(image_source):
                with open(image_source, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                 # Assume it's already base64 string if not file/url
                 return image_source
        return ""

    def __call__(self, text_prompt: str, image: Union[str, Image.Image], **kwargs) -> str:
        """Sends a vision-language request to the model."""
        
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)

        if self.provider == "openai":
            base64_image = self._encode_image(image)
            image_content = {}
            if isinstance(image, str) and image.startswith("http"):
                 image_content = {"type": "image_url", "image_url": {"url": image}}
            else:
                 image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            image_content,
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content

        elif self.provider == "anthropic":

            raise NotImplementedError("VLMClient does not support Anthropic yet due to differences in image handling and API structure.")   
        
            base64_image = self._encode_image(image)
            # Anthropic needs media_type, assuming jpeg for simplicity or detect
            media_type = "image/jpeg"
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": text_prompt}
                        ],
                    }
                ],
            )
            self._update_metrics(response.usage.input_tokens, response.usage.output_tokens)
            return response.content[0].text
            
        elif self.provider == "groq":
            # Groq VLM accepts either remote URLs or inline base64 image data URLs.
            base64_image = self._encode_image(image)

            if isinstance(image, str) and image.startswith("http"):
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": image
                    }
                }
            elif isinstance(image, str) and os.path.isfile(image):
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            else:
                raise ValueError("Groq provider currently only supports image URLs, local image files, PIL images, or base64-encoded image strings.")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text_prompt
                            },
                            image_content,
                        ]
                    }
                ],
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            if hasattr(response, 'usage'):
                self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content

        elif self.provider == "gemini":

            raise NotImplementedError("VLMClient does not support Gemini yet due to differences in image handling and API structure.")
        
            # Gemini supports PIL images directly or bytes
            if isinstance(image, str):
                if image.startswith("http"):
                    # quick download for gemini
                    # genai SDK might handle urls if passed as Part/URI, but keeping it simple with requests
                    if requests:
                        response = requests.get(image)
                        img_data = Image.open(BytesIO(response.content))
                    else:
                        raise ImportError("Requests not installed for URL handling")
                elif os.path.isfile(image):
                    img_data = Image.open(image)
                else: 
                     # Base64 string
                     img_data = Image.open(BytesIO(base64.b64decode(image)))
            else:
                img_data = image

            config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[text_prompt, img_data],
                config=config
            )
            usage = response.usage_metadata
            self._update_metrics(usage.prompt_token_count, usage.candidates_token_count)
            return response.text

        else:
             raise NotImplementedError(f"Provider {self.provider} not supported for Vision.")


if __name__ == "__main__":

    from src.test.test_client import TestVLMClient

    model_parameters = {
        "model_name": "groq/llama4-scout-17b",
        'temperature': 0.5,
        'max_tokens': 512,
        'top_p': 0.9
    }

    vlm_client = TestVLMClient(**model_parameters)
    vlm_client.test_response_with_url_image()
    vlm_client.test_response_with_local_image()