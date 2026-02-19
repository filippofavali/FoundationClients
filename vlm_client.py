import os
import base64
from io import BytesIO
from typing import Union

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from base_client import BaseFoundationClient

class VLMClient(BaseFoundationClient):
    """
    Client for Vision-Language tasks.
    """
    
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
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
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
             # Similar to OpenAI for Llama 3.2 vision models
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
            if hasattr(response, 'usage'):
                self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content

        elif self.provider == "gemini":
            # Gemini supports PIL images directly
            if isinstance(image, str):
                if image.startswith("http"):
                    # quick download for gemini
                    response = requests.get(image)
                    img_data = Image.open(BytesIO(response.content))
                elif os.path.isfile(image):
                    img_data = Image.open(image)
                else: 
                     # Base64 string
                     img_data = Image.open(BytesIO(base64.b64decode(image)))
            else:
                img_data = image

            config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            response = self.client.generate_content([text_prompt, img_data], generation_config=config)
            usage = response.usage_metadata
            self._update_metrics(usage.prompt_token_count, usage.candidates_token_count)
            return response.text

        else:
             raise NotImplementedError(f"Provider {self.provider} not supported for Vision.")
