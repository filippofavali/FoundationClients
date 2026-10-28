# FoundationClients

One interface for text (`LLMClient`) and vision (`VLMClient`) calls across Groq, Nebius, OpenAI and Anthropic.

## Setup

```bash
pip install -r requirements.txt
```

Put keys in `.env` (read automatically), named `<PROVIDER>_API_KEY`:

```
GROQ_API_KEY=...
NEBIUS_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

`api_key=` / `base_url=` passed to the constructor override the env (`NEBIUS_BASE_URL` defaults to `https://api.tokenfactory.nebius.com/v1/`).

## Creating a client

```python
from src.llm_client import LLMClient
from src.vlm_client import VLMClient

llm = LLMClient(model_name="anthropic/claude-opus-5.5", max_tokens=4096)
vlm = VLMClient(model_name="openai/gpt-5-mini")
```

- `model_name` is **required**, format `provider/model`. The model part is either an alias from `PROVIDER_MODELS` in [src/base_client.py](src/base_client.py) or any raw provider model ID (passed through unchanged).
- Constructor parameters are defaults for every call. Any of them can be overridden per call as a keyword argument.

| Parameter | Default | Notes |
|---|---|---|
| `temperature` | `0.7` | See [sampling rules](#sampling-rules) |
| `top_p` | `1.0` | See [sampling rules](#sampling-rules) |
| `max_tokens` | `1024` | Raise it for Anthropic Opus/Sonnet 5.5 and OpenAI gpt-5: their reasoning counts against it |
| `stream` | `False` | Prints tokens as they arrive and returns the full text |
| `full_content` | `False` | Return the raw SDK response object instead of the text (not available on every path, see matrix) |

## LLMClient

```python
llm(user_message=None, system_message="You are a helpful assistant.",
    force_json=False, forced_json_schema=None, **kwargs) -> str
```

Two ways to send input:

```python
# 1. Single turn
llm(user_message="What is the capital of Italy?", system_message="Answer in one word.")

# 2. Full conversation (OpenAI-style messages; user_message/system_message are ignored)
llm(messages=[
    {"role": "system", "content": "You are a chemistry expert."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Formula of table salt?"},
])
```

For Anthropic, `system` messages in the list are removed from `messages` and sent as the top-level `system` field.

## VLMClient

```python
vlm(text_prompt=None, image=None, force_json=False, forced_json_schema=None, **kwargs) -> str
```

`image` accepts:

| Type | Example |
|---|---|
| Local path | `"test_images/test_silvio.jpg"` |
| URL (must point to an image file, not an HTML page) | `"https://.../photo.png"` |
| `PIL.Image.Image` | `Image.open(...)` (re-encoded as JPEG) |
| `bytes` | raw file bytes |
| base64 string | already-encoded image data |

The image's MIME type is detected from the file (override with `image_mime_type="image/png"`). One image per call. Use `messages=` (OpenAI/Nebius/Anthropic only) to send several images or several turns.

Other VLM options: `image_detail="low"|"high"` (OpenAI/Nebius), `system_message=` (Anthropic only).

Helper: `vlm._draw_bbs(boxes, image, print=True)` draws `[{label, x_min, y_min, x_max, y_max}]` boxes on the image.

## Structured output

Define the expected output as a **Pydantic model**. Pass it with `force_json=True`:

```python
from pydantic import BaseModel
import json

class KeyEntity(BaseModel):
    entity: str
    type: str

class EmailClassification(BaseModel):
    category: str
    priority: str
    key_entities: list[KeyEntity]
    requires_immediate_attention: bool

raw = llm(user_message="Classify: 'Server is down!'", force_json=True,
          forced_json_schema=EmailClassification)
result = EmailClassification.model_validate(json.loads(raw))
```

- The client returns a **JSON string**. Validate it yourself with `model_validate(json.loads(...))`.
- The schema is made strict before it is sent: every object gets `additionalProperties: false` and all fields become required. Optional fields are therefore not supported. Use `X | None` if a value may be missing.
- Nested models, lists and enums work. Avoid recursive models.
- For Anthropic, numeric/string/array constraints (`Field(ge=..., max_length=...)`) are removed from the schema. Validate them client-side.

How it maps per provider:

| | `force_json=True` + `forced_json_schema` | `force_json=True` alone |
|---|---|---|
| Groq | `response_format: json_schema` (strict) | LLM: `json_object`. VLM: **ignored** (put the JSON format in the prompt) |
| Nebius | `response_format: json_schema` (strict) | `json_object` |
| OpenAI | `response_format: json_schema` (strict) | `json_object` (the prompt must mention "JSON") |
| Anthropic | `output_config.format: json_schema` | A "respond only with JSON" instruction is appended to the system prompt |

`response_format=` passed explicitly (OpenAI/Nebius) takes precedence over `force_json`.

## Capability matrix

| | Groq | Nebius | OpenAI | Anthropic |
|---|---|---|---|---|
| LLM `messages=` | ✓ | ✓ | ✓ | ✓ |
| LLM `stream` / `full_content` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| VLM `messages=` | ✗ | ✓ | ✓ | ✓ |
| VLM `stream` / `full_content` | ✗ / ✗ | ✓ / ✗ | ✓ / ✗ | ✓ / ✓ |

Gemini clients are not implemented (`NotImplementedError`).

**Extra kwargs** (on the constructor or per call). Groq passes none beyond the common parameters.

- **OpenAI / Nebius:** `n`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`, `top_logprobs`, `user`, `seed`, `stream_options`, `response_format`, `reasoning_effort`. `top_k`, `guided_json` and `extra_body` go in `extra_body` (useful on Nebius/vLLM).
- **Anthropic:** `effort` (`"low"` to `"max"`, the main quality/cost control on 5.x models), `stop_sequences`, `top_k`, `thinking`, `metadata`, `service_tier`, `output_config`, `fallbacks`.

## Sampling rules

- **Groq / Nebius:** `temperature` and `top_p` are always sent.
- **OpenAI:** sent for gpt-4o/gpt-4.1. **Dropped** for reasoning models (`gpt-5*`, `o1`/`o3`/`o4`), which reject them. Use `reasoning_effort` there. OpenAI uses `max_completion_tokens`, set from `max_tokens`.
- **Anthropic:** dropped for Opus/Sonnet 5.x, Fable and Opus 4.7/4.8, which reject them. Use `effort` there. On older models (e.g. Haiku 4.5) only the one **you set explicitly** is sent: `temperature` if given, otherwise `top_p`.

## Anthropic specifics

- **Thinking:** Opus 5.5 always thinks first. Only the text blocks are returned. Thinking tokens count toward `max_tokens`.
- **Refusal fallback:** on Opus 5.5, Sonnet 5.5, Opus 5 and Fable 5.1, `fallbacks="default"` is on by default. It retries a refused request on another model server-side. Disable it with `fallbacks=None`.
- **Warnings:** a refusal or an answer truncated at `max_tokens` prints a warning. The (possibly partial) text is still returned.

## Usage tracking

Every non-streaming call (and Anthropic streaming calls) records token usage:

```python
llm.get_total_usage()   # {'total_tokens': ..., 'input_tokens': ..., 'output_tokens': ...}
llm.log_metrics()       # per-call DataFrame
llm.usage_metrics       # pandas DataFrame or None
```

## Live examples

```bash
python example_of_usage.py --llm --vlm --anthropic --groq
```

- **Suites:** `--llm` (open question, structured email classification) and `--vlm` (image question, structured scene description, face bounding boxes, robot grasp affordances).
- **Providers:** `--openai --anthropic --groq --nebius --google`. Each selected suite runs on each selected provider. No suite flag runs both, and no provider flag runs all five.
- **Models:** set per provider in `MODELS` at the top of the script.
- **`--show`:** opens the images annotated with boxes and grasp points.
- **Skipped tests:** a provider without a key, Google (not implemented), and the affordance test until `test_images/robotic_affordances.jpg` exists.
- **Output:** a PASS/FAIL/SKIP summary with time and tokens per test. The exit code is non-zero if anything failed.
