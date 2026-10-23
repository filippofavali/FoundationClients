import os
from typing import List, Dict, Union, Optional, Any
from abc import ABC, abstractmethod

import pandas as pd
from dotenv import load_dotenv  # type: ignore[reportMissingImports]

# Import provider SDKs
try:
    from groq import Groq  # type: ignore[reportMissingImports]
except ImportError:
    Groq = None

try:
    from openai import OpenAI  # type: ignore[reportMissingImports]
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic  # type: ignore[reportMissingImports]
except ImportError:
    Anthropic = None

try:
    from google import genai
    from google.genai import types  # type: ignore[reportMissingImports]
except ImportError:
    genai = None

load_dotenv()


# OpenAI reasoning models reject non-default temperature/top_p and the legacy max_tokens parameter.
OPENAI_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Anthropic models on which temperature/top_p/top_k were removed (non-default values return a 400).
ANTHROPIC_NO_SAMPLING_MODEL_PREFIXES = (
    "claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5",
    "claude-opus-4-7", "claude-opus-4-8",
)

# Anthropic models that accept the server-side refusal fallback (`fallbacks: "default"`).
ANTHROPIC_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")

# JSON schema keywords that Anthropic structured outputs do not support.
ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
)


def strict_json_schema(schema_model, drop_keys: tuple = ()) -> Dict[str, Any]:
    """
    Builds a strict JSON schema from a Pydantic model: every object gets
    additionalProperties=false and all of its properties marked as required.
    """
    schema_dict = schema_model.model_json_schema()

    def enforce_strict_mode(schema_node):
        if isinstance(schema_node, dict):
            for key in drop_keys:
                schema_node.pop(key, None)
            if (schema_node.get("type") == "object" or "properties" in schema_node) and "additionalProperties" not in schema_node:
                schema_node["additionalProperties"] = False

            if "properties" in schema_node:
                schema_node["required"] = list(schema_node["properties"].keys())

            for val in schema_node.values():
                enforce_strict_mode(val)
        elif isinstance(schema_node, list):
            for item in schema_node:
                enforce_strict_mode(item)

    enforce_strict_mode(schema_dict)
    return schema_dict


def openai_json_schema_response_format(schema_model) -> Dict[str, Any]:
    """response_format payload for OpenAI-compatible strict structured outputs."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_model.__name__,
            "strict": True,
            "schema": strict_json_schema(schema_model),
        },
    }


class ModelRegistry:
    """
    Centralized registry for model names across providers.
    Maps simplified/internal names to official API model IDs.
    """
    
    # Groq Models
    GROQ_MODELS = {
        "openai-oss-20b": "openai/gpt-oss-20b",
        "openai-oss-120b": "openai/gpt-oss-120b",
        "qwen3-32b": "qwen/qwen3-32b",
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        # vision enabled models
        "qwen3.6-27b": "qwen/qwen3.6-27b",
        "qwen3.8-27b": "qwen/Qwen3.8-27B"
    }

    # Nebius Models
    NEBIUS_MODELS = {
        'google-gemma-27b': 'google/gemma-3-27b-it',
        'nvidia-nemotron-30b': "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        'nvidia-nemotron-120b': "nvidia/nemotron-3-super-120b-a12b",
        'nvidia-cosmos3-33b': "nvidia/Cosmos3-Super-Reasoner",
        'kimi-k2.6': "moonshotai/Kimi-K2.6",
        'qwen3-embedding-8b': 'Qwen/Qwen3-Embedding-8B',
        # vision enabled models
        'qwen3-2.5-70b': 'Qwen/Qwen2.5-VL-72B-Instruct',
        "zAI-GLM-5.3": "zai-org/GLM-5.3-Flash",
    }

    # OpenAI Models
    OPENAI_MODELS = {
        "gpt-5": "gpt-5",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        # Vision is supported by gpt-5*, gpt-4.1* and gpt-4o*
    }

    # Anthropic Models (all current models accept image input)
    ANTHROPIC_MODELS = {
        "claude-fable-5.1": "claude-fable-5-1",
        "claude-opus-5.5": "claude-opus-5-5",
        "claude-sonnet-5.5": "claude-sonnet-5-5",
        "claude-haiku-4.5": "claude-haiku-4-5",
    }

    # Gemini Models
    GEMINI_MODELS = {
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.0-pro": "gemini-1.0-pro",
    }

    @classmethod
    def get_model_id(cls, provider: str, model_name: str) -> str:
        """Resolves the internal model name to the provider's API model ID."""
        if provider == "groq":
            return cls.GROQ_MODELS.get(model_name, model_name)
        elif provider == "openai":
            return cls.OPENAI_MODELS.get(model_name, model_name)
        elif provider == "nebius":
            return cls.NEBIUS_MODELS.get(model_name, model_name)
        elif provider == "anthropic":
            return cls.ANTHROPIC_MODELS.get(model_name, model_name)
        elif provider == "gemini":
            return cls.GEMINI_MODELS.get(model_name, model_name)
        else:
            return model_name


class BaseFoundationClient(ABC):
    """Base class for all foundation model clients."""
    
    def __init__(self, **model_parameters):
        
        self.model_parameters = model_parameters
        full_model_name = model_parameters.get("model_name", "")
        if "/" in full_model_name:
            self.provider, self.raw_model_name = full_model_name.split("/", 1)
        else:
            raise ValueError(f"Model name '{full_model_name}' must be in format 'provider/model_name'")

        self.model_name = ModelRegistry.get_model_id(self.provider, self.raw_model_name)
        
        self.temperature = model_parameters.get("temperature", 0.7)
        self.max_tokens = model_parameters.get("max_tokens", 1024)
        self.top_p = model_parameters.get("top_p", 1.0)
        self.stream = model_parameters.get("stream", False)
        
        self.api_key = model_parameters.get("api_key", os.getenv(f"{self.provider.upper()}_API_KEY"))
        self.base_url = model_parameters.get("base_url", os.getenv(f"{self.provider.upper()}_BASE_URL"))
        
        self.client = self._initialize_client()
        self.usage_metrics = None

    def _initialize_client(self):
        if self.provider == "groq":
            if not Groq: raise ImportError("Groq SDK not installed.")
            return Groq(api_key=self.api_key)
        elif self.provider == "openai":
            if not OpenAI: raise ImportError("OpenAI SDK not installed.")
            return OpenAI(api_key=self.api_key)
        elif self.provider == "nebius":
            if not OpenAI: raise ImportError("OpenAI SDK not installed.")
            base_url = self.base_url or "https://api.tokenfactory.nebius.com/v1/"
            return OpenAI(api_key=self.api_key, base_url=base_url)
        elif self.provider == "anthropic":
            if not Anthropic: raise ImportError("Anthropic SDK not installed.")
            return Anthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            if not genai: raise ImportError("Google GenAI SDK not installed.")
            # Initialize Client directly
            return genai.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _is_explicit_parameter(self, name: str, kwargs: Dict[str, Any]) -> bool:
        """True when the caller set the parameter on the call or at client construction."""
        return name in kwargs or name in self.model_parameters

    def _call_anthropic_messages(self, messages: list, system_message: Optional[str], force_json: bool = False, forced_json_schema=None, **kwargs):
        """
        Shared Anthropic Messages API call used by both LLM and VLM clients.
        `messages` must already be in Anthropic format (no system-role entries).
        """
        stream = kwargs.get("stream", self.stream)
        full_content = kwargs.get("full_content", False)

        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        system_prompt = system_message or ""
        if force_json and forced_json_schema is None:
            # Anthropic has no json_object mode: enforce plain JSON via the system prompt.
            system_prompt = (system_prompt + "\n\nRespond only with a single valid JSON object, without markdown fences or extra text.").strip()
        if system_prompt:
            params["system"] = system_prompt

        # Sampling parameters: removed on the newest models, and older models accept only one of temperature/top_p.
        if not self.model_name.startswith(ANTHROPIC_NO_SAMPLING_MODEL_PREFIXES):
            if self._is_explicit_parameter("temperature", kwargs):
                params["temperature"] = kwargs.get("temperature", self.temperature)
            elif self._is_explicit_parameter("top_p", kwargs):
                params["top_p"] = kwargs.get("top_p", self.top_p)
            top_k = kwargs.get("top_k", self.model_parameters.get("top_k"))
            if top_k is not None:
                params["top_k"] = top_k

        for param_name in ["stop_sequences", "thinking", "metadata", "service_tier"]:
            value = kwargs.get(param_name, self.model_parameters.get(param_name))
            if value is not None:
                params[param_name] = value

        output_config = dict(kwargs.get("output_config", self.model_parameters.get("output_config")) or {})
        effort = kwargs.get("effort", self.model_parameters.get("effort"))
        if effort is not None:
            output_config["effort"] = effort
        if force_json and forced_json_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": strict_json_schema(forced_json_schema, drop_keys=ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS),
            }
        if output_config:
            params["output_config"] = output_config

        # Server-side refusal fallback, on by default for the models that support it; pass fallbacks=None to disable.
        default_fallbacks = "default" if self.model_name in ANTHROPIC_FALLBACK_MODELS else None
        fallbacks = kwargs.get("fallbacks", self.model_parameters.get("fallbacks", default_fallbacks))
        if fallbacks is not None:
            messages_api = self.client.beta.messages
            params["betas"] = ["server-side-fallback-2026-07-01" if fallbacks == "default" else "server-side-fallback-2026-06-01"]
            params["extra_body"] = {"fallbacks": fallbacks}
        else:
            messages_api = self.client.messages

        if stream:
            with messages_api.stream(**params) as response_stream:
                for text in response_stream.text_stream:
                    print(text, end="", flush=True)
                print()
                response = response_stream.get_final_message()
        else:
            response = messages_api.create(**params)

        if getattr(response, "usage", None) is not None:
            self._update_metrics(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal":
            print(f"[{self.__class__.__name__}] Warning: the model refused the request (stop_details={getattr(response, 'stop_details', None)}).")
        elif response.stop_reason == "max_tokens":
            print(f"[{self.__class__.__name__}] Warning: response truncated at max_tokens={params['max_tokens']}.")

        if full_content:
            return response
        return "".join(block.text for block in response.content if block.type == "text")

    def _update_metrics(self, input_tokens: int, output_tokens: int, search_provider: str = None):
        """Standardized metric collection."""
        new_metric = {
            "timestamp": pd.Timestamp.now(),
            "provider": self.provider,
            "model": self.model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if search_provider:
             new_metric["search_provider"] = search_provider
             
        if self.usage_metrics is None:
            self.usage_metrics = pd.DataFrame([new_metric])
        else:
            self.usage_metrics = pd.concat([self.usage_metrics, pd.DataFrame([new_metric])], ignore_index=True)

    def get_total_usage(self) -> Dict[str, int]:
        if self.usage_metrics is None:
             return {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        return {
            "total_tokens": int(self.usage_metrics["total_tokens"].sum()),
            "input_tokens": int(self.usage_metrics["input_tokens"].sum()),
            "output_tokens": int(self.usage_metrics["output_tokens"].sum()),
        }

    def log_metrics(self):
        if self.usage_metrics is not None:
             print(f"\n[{self.__class__.__name__}] Usage Metrics:")
             print(self.usage_metrics)
