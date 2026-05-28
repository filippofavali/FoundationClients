import os, sys
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .base_client import BaseFoundationClient
except ImportError:
    from base_client import BaseFoundationClient
try:
    from google import genai
except ImportError:
    genai = None

class LLMClient(BaseFoundationClient):
    """
    Client for Text-to-Text interaction.
    """

    def __init__(self, **model_parameters):
        super().__init__(**model_parameters)

    def _get_call_parameter(self, name: str, kwargs: Dict[str, Any], default: Any = None) -> Any:
        if name in kwargs:
            return kwargs[name]
        return self.model_parameters.get(name, default)
    
    def __call__(self, user_message: Optional[str] = None, system_message: str = "You are a helpful assistant.", **kwargs) -> str:
        # Merge call-specific overrides
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        stream = kwargs.get("stream", self.stream)
        full_content = kwargs.get("full_content", False)
        
        if self.provider in ["groq", "openai", "nebius"]:

            messages = kwargs.get("messages")
            if messages is None:
                messages = []
                if system_message:
                    messages.append({
                        "role": "system",
                        "content": system_message
                    })
                if user_message is not None:
                    messages.append({
                        "role": "user",
                        "content": user_message
                    })
            if not messages:
                raise ValueError("Either user_message or messages must be provided.")
            
            # OpenAI-compatible chat completion parameters.
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
            }
            
            if self.provider == "groq":
                 params["max_completion_tokens"] = max_tokens
            else:
                 params["max_tokens"] = max_tokens

            optional_params = [
                "n",
                "stream_options",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "logprobs",
                "top_logprobs",
                "user",
                "response_format",
            ]
            for param_name in optional_params:
                value = self._get_call_parameter(param_name, kwargs)
                if value is not None:
                    params[param_name] = value

            extra_body = self._get_call_parameter("extra_body", kwargs)
            guided_json = self._get_call_parameter("guided_json", kwargs)
            top_k = self._get_call_parameter("top_k", kwargs)
            if extra_body is not None:
                extra_body = dict(extra_body)
            elif guided_json is not None or top_k is not None:
                extra_body = {}
            if guided_json is not None:
                extra_body["guided_json"] = guided_json
            if top_k is not None:
                extra_body["top_k"] = top_k
            if extra_body is not None:
                params["extra_body"] = extra_body

            response = self.client.chat.completions.create(**params)
            
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end="", flush=True)
                print() # Newline
                return full_response
            
            elif full_content:
                if hasattr(response, 'usage'):
                    self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response
            
            else:
                content = response.choices[0].message.content
                if hasattr(response, 'usage'):
                    self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
                return content

        elif self.provider == "anthropic":

            raise NotImplementedError("LLMClient does not support Anthropic yet due to differences in system message handling.")
        
            # Anthropic does not support system message in the messages list in the same way (it's a top level param)
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            content = response.content[0].text
            self._update_metrics(response.usage.input_tokens, response.usage.output_tokens)
            return content

        elif self.provider == "gemini":

            raise NotImplementedError("LLMClient does not support Gemini yet due to differences in system message handling.")

            config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": top_p,
            }
            if system_message:
                config["system_instruction"] = system_message

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_message,
                config=config
            )
            
            usage = response.usage_metadata
            self._update_metrics(usage.prompt_token_count, usage.candidates_token_count)
            
            return response.text

        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented.")


if __name__ == "__main__":

    from src.test.test_client import TestLLMClient

    model_parameters = {
        "model_name": "groq/openai-oss-20b",
        'temperature': 1.5,
        'max_tokens': 2048,
        'top_p': 0.9
    }

    llm_client = TestLLMClient(
        **model_parameters
    )

    llm_client.test_response()
