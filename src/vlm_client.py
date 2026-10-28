import os, base64
from io import BytesIO
from typing import Any, Dict, Optional, Type, Union
from PIL import Image, ImageDraw
try:    
    from google import genai
except ImportError:
    genai = None
try:
    from .base_client import BaseFoundationClient, openai_json_schema_response_format
except ImportError:
    from src.base_client import BaseFoundationClient, openai_json_schema_response_format
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

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
            if image_source.mode not in ("RGB", "L"):
                image_source = image_source.convert("RGB")
            image_source.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image_source, bytes):
            return base64.b64encode(image_source).decode('utf-8')
        elif isinstance(image_source, str):
            if image_source.startswith("http"):
                return image_source
            elif os.path.isfile(image_source):
                with open(image_source, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                 return image_source
        return ""

    def _build_image_url_content(self, image: Union[str, bytes, Image.Image], **kwargs) -> Dict[str, Any]:
        if image is None:
            raise ValueError("image must be provided when messages are not passed.")

        image_url = {}
        if isinstance(image, str) and image.startswith("http"):
            image_url["url"] = image
        else:
            mime_type = self._detect_mime_type(image, **kwargs)
            image_url["url"] = f"data:{mime_type};base64,{self._encode_image(image)}"

        image_detail = kwargs.get("image_detail")
        if image_detail is not None:
            image_url["detail"] = image_detail

        return {
            "type": "image_url",
            "image_url": image_url
        }

    def _detect_mime_type(self, image: Union[str, bytes, Image.Image], **kwargs) -> str:
        """Media type of the encoded image (PIL images are always re-encoded as JPEG)."""
        if kwargs.get("image_mime_type"):
            return kwargs["image_mime_type"]
        if isinstance(image, Image.Image):
            return "image/jpeg"
        try:
            if isinstance(image, bytes):
                image_format = Image.open(BytesIO(image)).format
            elif isinstance(image, str) and os.path.isfile(image):
                with Image.open(image) as opened:
                    image_format = opened.format
            else:
                image_format = None
        except Exception:
            image_format = None
        return Image.MIME.get(image_format, "image/jpeg") if image_format else "image/jpeg"

    def _build_anthropic_image_content(self, image: Union[str, bytes, Image.Image], **kwargs) -> Dict[str, Any]:
        if image is None:
            raise ValueError("image must be provided when messages are not passed.")

        if isinstance(image, str) and image.startswith("http"):
            source = {"type": "url", "url": image}
        else:
            source = {
                "type": "base64",
                "media_type": self._detect_mime_type(image, **kwargs),
                "data": self._encode_image(image),
            }
        return {"type": "image", "source": source}

    def _draw_bbs(self, bbs: list, image: Union[str, Image.Image], print: bool = False):
        if isinstance(image, str):
            image = Image.open(image)

        image = image.copy()
        draw = ImageDraw.Draw(image)

        for bb in bbs:
            coordinates = bb.get("coordinates", bb)
            x_min = coordinates["x_min"]
            y_min = coordinates["y_min"]
            x_max = coordinates["x_max"]
            y_max = coordinates["y_max"]
            label = bb.get("label", "")

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            if label:
                draw.text((x_min, max(0, y_min - 12)), label, fill="red")

        if print:
            image.show()
            return

        return image

    def _call_groq(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        base64_image = self._encode_image(image)

        if isinstance(image, str) and image.startswith("http"):
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image
                }
            }
        else:
            # base64_image covers local files, bytes, PIL images, and raw base64 strings
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }

        params = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        image_content,
                    ]
                }
            ],
            **self._collect_parameters(kwargs),
        }
        if force_json and forced_json_schema is not None and "response_format" not in params:
            params["response_format"] = openai_json_schema_response_format(forced_json_schema)
        # Note: Groq VLMs (e.g. qwen 3.6) don't support response_format json_object.
        # JSON output is enforced via the prompt instead.

        response = self.client.chat.completions.create(**params)
        if hasattr(response, 'usage'):
            self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content

    def _call_openai(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        stream = kwargs.get("stream", self.stream)

        messages = kwargs.get("messages")
        if messages is None:
            if text_prompt is None:
                raise ValueError("text_prompt must be provided when messages are not passed.")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        self._build_image_url_content(image, **kwargs),
                    ],
                }
            ]

        params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **self._collect_parameters(kwargs),
        }

        if force_json and "response_format" not in params:
            if forced_json_schema is not None:
                params["response_format"] = openai_json_schema_response_format(forced_json_schema)
            else:
                params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**params)
        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
            print()
            return full_response

        if hasattr(response, 'usage'):
            self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content

    def _call_nebius(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        return self._call_openai(text_prompt, image, force_json=force_json, forced_json_schema=forced_json_schema, **kwargs)

    def _call_inception(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], **kwargs) -> str:
        raise NotImplementedError("VLMClient does not support Inception: no Mercury chat model accepts image input.")

    def _call_anthropic(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        system_message = kwargs.pop("system_message", None)
        messages = kwargs.pop("messages", None)
        if messages is None:
            if text_prompt is None:
                raise ValueError("text_prompt must be provided when messages are not passed.")
            # Images before text is the recommended ordering for Claude.
            messages = [
                {
                    "role": "user",
                    "content": [
                        self._build_anthropic_image_content(image, **kwargs),
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
        return self._call_anthropic_messages(messages, system_message, force_json=force_json, forced_json_schema=forced_json_schema, **kwargs)

    def _call_gemini(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], **kwargs) -> str:
        raise NotImplementedError("VLMClient does not support Gemini yet due to differences in image handling and API structure.")

    def __call__(self, text_prompt: Optional[str] = None, image: Union[str, bytes, Image.Image, None] = None, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        """Sends a vision-language request to the model."""
        if self.provider == "groq":
            return self._call_groq(text_prompt, image, force_json=force_json, forced_json_schema=forced_json_schema, **kwargs)
        elif self.provider == "openai":
            return self._call_openai(text_prompt, image, force_json=force_json, forced_json_schema=forced_json_schema, **kwargs)
        elif self.provider == "nebius":
            return self._call_nebius(text_prompt, image, force_json=force_json, forced_json_schema=forced_json_schema, **kwargs)
        elif self.provider == "inception":
            return self._call_inception(text_prompt, image, **kwargs)
        elif self.provider == "anthropic":
            return self._call_anthropic(text_prompt, image, force_json=force_json, forced_json_schema=forced_json_schema, **kwargs)
        elif self.provider == "gemini":
            return self._call_gemini(text_prompt, image, **kwargs)
        else:
             raise NotImplementedError(f"Provider {self.provider} not supported for Vision.")


if __name__ == "__main__":
        
        use_nebius = True
        use_groq = False

        if use_nebius:
            model_parameters = {
                "model_name": "nebius/qwen3-2.5-70b",
                'temperature': 0.7,
                'max_tokens': 2048,
                'top_p': 0.9,
            }
        elif use_groq:
            model_parameters = {
                "model_name": "groq/llama4-scout-17b",
                'temperature': 0.7,
                'max_tokens': 2048,
                'top_p': 0.9
            }
             
        vlm = VLMClient(**model_parameters)
        task = 'Find all the faces in the image. If there are specific known, please label them with their names.'
        
        bb_prompt = """
        Task: {task}.
        The image is provided in the size of {pixels_width} x {pixels_height}.
        Strictly use the following json format for the response, avoid any additional text or explanation.

        {{
        "bounding_boxes": [
            {{
                "label": "detection-label",
                "x_min": top-left-x-pixel,
                "y_min": top-left-y-pixel,
                "x_max": bottom-right-x-pixel,
                "y_max": bottom-right-y-pixel
            }}, 
            ]
        }}
        """
