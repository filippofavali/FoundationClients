import os
from typing import List, Dict, Union, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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


# =====================================================================================
# 1. PROVIDER PARAMETERS
# Every request parameter a provider's API accepts. Anything outside these sets is dropped
# (with a warning) before the request is sent.
#   accepted              -> parameters forwarded to the API
#   renamed               -> internal name -> name the API expects
#   extra_body            -> accepted parameters sent inside `extra_body` (non-standard fields)
#   send_default_sampling -> send the client's default temperature/top_p even when not set explicitly
# =====================================================================================
OPENAI_COMPATIBLE_PARAMETERS = frozenset({
    "temperature", "top_p", "max_tokens", "n", "stop", "stream_options",
    "presence_penalty", "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
    "user", "seed", "response_format", "reasoning_effort",
})


@dataclass(frozen=True)
class ProviderParameters:
    accepted: frozenset
    renamed: Dict[str, str] = field(default_factory=dict)
    extra_body: frozenset = frozenset()
    send_default_sampling: bool = True


PROVIDER_PARAMETERS: Dict[str, ProviderParameters] = {
    "openai": ProviderParameters(
        accepted=OPENAI_COMPATIBLE_PARAMETERS,
        renamed={"max_tokens": "max_completion_tokens"},
    ),
    "nebius": ProviderParameters(
        accepted=OPENAI_COMPATIBLE_PARAMETERS | {"top_k", "guided_json"},
        extra_body=frozenset({"top_k", "guided_json"}),
    ),
    "groq": ProviderParameters(
        accepted=frozenset({
            "temperature", "top_p", "max_tokens", "n", "stop", "seed", "user",
            "presence_penalty", "frequency_penalty", "response_format", "service_tier",
            "reasoning_effort", "reasoning_format", "include_reasoning",
        }),
        renamed={"max_tokens": "max_completion_tokens"},
    ),
    "anthropic": ProviderParameters(
        accepted=frozenset({
            "temperature", "top_p", "top_k", "max_tokens", "stop_sequences", "thinking",
            "metadata", "service_tier", "output_config", "effort", "fallbacks",
        }),
        send_default_sampling=False,
    ),
    # Placeholder: Gemini is not wired into the clients yet.
    "gemini": ProviderParameters(accepted=frozenset({"temperature", "top_p", "max_tokens"})),
}

# Arguments consumed by the clients themselves: never forwarded as-is, never warned about.
CLIENT_PARAMETERS = frozenset({
    "model_name", "api_key", "base_url", "stream", "full_content", "messages",
    "system_message", "image_detail", "image_mime_type", "extra_body",
})


# =====================================================================================
# 2. MODEL REGISTRY
# Simplified/internal names -> official API model IDs, one block per provider.
# =====================================================================================
PROVIDER_MODELS: Dict[str, Dict[str, str]] = {
    "groq": {
        "openai-oss-20b": "openai/gpt-oss-20b",
        "openai-oss-120b": "openai/gpt-oss-120b",
        "qwen3-32b": "qwen/qwen3-32b",
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        # vision enabled models
        "qwen3.6-27b": "qwen/qwen3.6-27b",
        "qwen3.8-27b": "qwen/Qwen3.8-27B",
    },
    "nebius": {
        "google-gemma-27b": "google/gemma-3-27b-it",
        "nvidia-nemotron-30b": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "nvidia-nemotron-120b": "nvidia/nemotron-3-super-120b-a12b",
        "nvidia-cosmos3-33b": "nvidia/Cosmos3-Super-Reasoner",
        "kimi-k2.6": "moonshotai/Kimi-K2.6",
        "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
        # vision enabled models
        "qwen3-2.5-70b": "Qwen/Qwen2.5-VL-72B-Instruct",
        "zAI-GLM-5.3": "zai-org/GLM-5.3-Flash",
    },
    "openai": {
        # Vision is supported by gpt-5*, gpt-4.1* and gpt-4o*
        "gpt-5": "gpt-5",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
    },
    "anthropic": {
        # All current models accept image input
        "claude-fable-5.1": "claude-fable-5-1",
        "claude-opus-5.5": "claude-opus-5-5",
        "claude-sonnet-5.5": "claude-sonnet-5-5",
        "claude-haiku-4.5": "claude-haiku-4-5",
    },
    "gemini": {
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.0-pro": "gemini-1.0-pro",
    },
}


class ModelRegistry:
    """Resolves internal model names to the provider's API model ID (see PROVIDER_MODELS)."""

    @classmethod
    def get_model_id(cls, provider: str, model_name: str) -> str:
        return PROVIDER_MODELS.get(provider, {}).get(model_name, model_name)


# =====================================================================================
# 3. MODEL-SPECIFIC OVERRIDES
# Exceptions for model families, applied on top of PROVIDER_PARAMETERS.
#   prefixes    -> matched with model_id.startswith
#   models      -> matched exactly
#   unsupported -> removed from the provider's accepted parameters
#   defaults    -> used when the caller does not set the parameter
# =====================================================================================
@dataclass(frozen=True)
class ModelRule:
    prefixes: tuple = ()
    models: tuple = ()
    unsupported: frozenset = frozenset()
    defaults: Dict[str, Any] = field(default_factory=dict)


MODEL_RULES: Dict[str, List[ModelRule]] = {
    "openai": [
        # Reasoning models reject non-default temperature/top_p.
        ModelRule(prefixes=("gpt-5", "o1", "o3", "o4"), unsupported=frozenset({"temperature", "top_p"})),
    ],
    "anthropic": [
        # temperature/top_p/top_k were removed on these models (non-default values return a 400).
        ModelRule(
            prefixes=("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5",
                      "claude-opus-4-7", "claude-opus-4-8"),
            unsupported=frozenset({"temperature", "top_p", "top_k"}),
        ),
        # Server-side refusal fallback on by default; pass fallbacks=None to disable.
        ModelRule(
            models=("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"),
            defaults={"fallbacks": "default"},
        ),
    ],
}


@dataclass(frozen=True)
class ResolvedParameters:
    """Provider parameters with the matching model rules applied."""
    accepted: frozenset
    renamed: Dict[str, str]
    extra_body: frozenset
    send_default_sampling: bool
    defaults: Dict[str, Any]


def resolve_parameters(provider: str, model_id: str) -> ResolvedParameters:
    spec = PROVIDER_PARAMETERS.get(provider, ProviderParameters(accepted=frozenset()))
    accepted, defaults = set(spec.accepted), {}
    for rule in MODEL_RULES.get(provider, []):
        if model_id in rule.models or model_id.startswith(rule.prefixes):
            accepted -= rule.unsupported
            defaults.update(rule.defaults)
    return ResolvedParameters(frozenset(accepted), spec.renamed, spec.extra_body, spec.send_default_sampling, defaults)


# =====================================================================================
# 4. OTHER PROVIDER CONFIG
# =====================================================================================
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
        self.parameters = resolve_parameters(self.provider, self.model_name)
        self._warned_parameters = set()
        
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

    def _collect_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds the provider request parameters from model rule defaults < client defaults
        < constructor parameters < call kwargs, keeping only what the provider/model accepts.
        Unsupported parameters set by the caller are dropped with a one-time warning.
        """
        spec = self.parameters
        explicit = set(self.model_parameters) | set(kwargs)

        merged = dict(spec.defaults)
        merged["max_tokens"] = self.max_tokens
        if spec.send_default_sampling:
            merged.update(temperature=self.temperature, top_p=self.top_p)
        merged.update(self.model_parameters)
        merged.update(kwargs)

        params = {}
        extra_body = dict(merged.get("extra_body") or {})
        for name, value in merged.items():
            if name in CLIENT_PARAMETERS:
                continue
            if name not in spec.accepted:
                if name in explicit and name not in self._warned_parameters:
                    self._warned_parameters.add(name)
                    print(f"[{self.__class__.__name__}] Warning: '{name}' is not supported by {self.provider}/{self.model_name} and was dropped.")
                continue
            if value is None:
                continue
            if name in spec.extra_body:
                extra_body[name] = value
            else:
                params[spec.renamed.get(name, name)] = value
        if extra_body:
            params["extra_body"] = extra_body
        return params

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
            **self._collect_parameters(kwargs),
        }

        system_prompt = system_message or ""
        if force_json and forced_json_schema is None:
            # Anthropic has no json_object mode: enforce plain JSON via the system prompt.
            system_prompt = (system_prompt + "\n\nRespond only with a single valid JSON object, without markdown fences or extra text.").strip()
        if system_prompt:
            params["system"] = system_prompt

        # Older models accept only one of temperature/top_p (unsupported models already dropped both).
        if "temperature" in params:
            params.pop("top_p", None)

        output_config = dict(params.pop("output_config", None) or {})
        effort = params.pop("effort", None)
        if effort is not None:
            output_config["effort"] = effort
        if force_json and forced_json_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": strict_json_schema(forced_json_schema, drop_keys=ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS),
            }
        if output_config:
            params["output_config"] = output_config

        # Server-side refusal fallback: default comes from MODEL_RULES; pass fallbacks=None to disable.
        fallbacks = params.pop("fallbacks", None)
        if fallbacks is not None:
            messages_api = self.client.beta.messages
            params["betas"] = ["server-side-fallback-2026-07-01" if fallbacks == "default" else "server-side-fallback-2026-06-01"]
            params["extra_body"] = {**params.get("extra_body", {}), "fallbacks": fallbacks}
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
