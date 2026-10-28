from typing import Any, Dict, Optional, Type
try:
    from .base_client import BaseFoundationClient, openai_json_schema_response_format
except ImportError:
    from src.base_client import BaseFoundationClient, openai_json_schema_response_format
try:
    from google import genai
except ImportError:
    genai = None
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None


class LLMClient(BaseFoundationClient):
    """
    Client for Text-to-Text interaction.
    """

    def __init__(self, **model_parameters):
        super().__init__(**model_parameters)

    def _build_messages(self, user_message: Optional[str], system_message: str, messages: Optional[list]) -> list:
        if messages is not None:
            return messages

        built_messages = []
        if system_message:
            built_messages.append({"role": "system", "content": system_message})
        if user_message is not None:
            built_messages.append({"role": "user", "content": user_message})
        return built_messages

    def _handle_stream(self, response):
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
        print()
        return full_response

    def _update_from_openai_usage(self, response):
        if hasattr(response, "usage"):
            self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)

    def _json_response_format(self, force_json: bool, forced_json_schema) -> Optional[Dict[str, Any]]:
        if force_json and forced_json_schema is not None:
            return openai_json_schema_response_format(forced_json_schema)
        if force_json:
            return {"type": "json_object"}
        return None

    def _call_openai_compatible(self, user_message: Optional[str], system_message: str, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        """Chat Completions call shared by Groq, OpenAI and Nebius (parameters filtered per provider)."""
        stream = kwargs.get("stream", self.stream)
        full_content = kwargs.get("full_content", False)
        messages = self._build_messages(user_message, system_message, kwargs.get("messages"))

        if not messages:
            raise ValueError("Either user_message or messages must be provided.")

        params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **self._collect_parameters(kwargs),
        }
        # An explicit response_format wins over force_json.
        json_format = self._json_response_format(force_json, forced_json_schema)
        if json_format is not None and "response_format" not in params:
            params["response_format"] = json_format

        response = self.client.chat.completions.create(**params)
        if stream:
            return self._handle_stream(response)
        self._update_from_openai_usage(response)
        if full_content:
            return response
        return response.choices[0].message.content

    def _call_groq(self, user_message: Optional[str], system_message: str, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        return self._call_openai_compatible(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)

    def _call_openai(self, user_message: Optional[str], system_message: str, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        return self._call_openai_compatible(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)

    def _call_nebius(self, user_message: Optional[str], system_message: str, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        return self._call_openai_compatible(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)

    def _call_inception(self, user_message: Optional[str], system_message: str, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        return self._call_openai_compatible(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)

    def _call_anthropic(self, user_message: Optional[str], system_message: str, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        messages = kwargs.pop("messages", None)
        if messages is None:
            if user_message is None:
                raise ValueError("Either user_message or messages must be provided.")
            messages = [{"role": "user", "content": user_message}]
        else:
            # Anthropic takes the system prompt as a top-level parameter, not as a message.
            system_parts = [m["content"] for m in messages if m.get("role") == "system" and isinstance(m.get("content"), str)]
            messages = [m for m in messages if m.get("role") != "system"]
            if system_parts:
                system_message = "\n\n".join(system_parts)

        return self._call_anthropic_messages(messages, system_message, force_json=force_json, forced_json_schema=forced_json_schema, **kwargs)

    def _call_gemini(self, user_message: Optional[str], system_message: str, force_json: bool = False, **kwargs) -> str:
        raise NotImplementedError("LLMClient does not support Gemini yet due to differences in system message handling.")

    def __call__(self, user_message: Optional[str] = None, system_message: str = "You are a helpful assistant.", force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        if self.provider == "groq":
            return self._call_groq(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)
        if self.provider == "openai":
            return self._call_openai(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)
        if self.provider == "nebius":
            return self._call_nebius(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)
        if self.provider == "inception":
            return self._call_inception(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)
        if self.provider == "anthropic":
            return self._call_anthropic(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)
        if self.provider == "gemini":
            return self._call_gemini(user_message, system_message, force_json, **kwargs)
        raise NotImplementedError(f"Provider {self.provider} not implemented.")


if __name__ == "__main__":

    from src.test.test_client import TestLLMClient  # type: ignore[reportMissingImports]

    model_parameters = {
        "model_name": "Nebius/openai-oss-20b",
        'temperature': 1.5,
        'max_tokens': 2048,
        'top_p': 0.9
    }

    llm_client = TestLLMClient(
        **model_parameters
    )

    llm_client.test_response()
