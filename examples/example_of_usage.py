"""
Live examples of the LLM and VLM clients across providers. This file is meant to be run as a script,
with <PROVIDER>_API_KEY set in .env (Google uses GEMINI_API_KEY).

    python example_of_usage.py                          # every suite on every provider
    python example_of_usage.py --llm --anthropic --groq # LLM suite on Anthropic and Groq
    python example_of_usage.py --vlm --openai --show    # VLM suite on OpenAI, opening annotated images

Every selected suite (--llm, --vlm) runs on every selected provider. No suite flag selects both;
no provider flag selects all of them.
"""

import os
import sys
import json
import time
import argparse
from traceback import format_exc

from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

try:
    from .src.llm_client import LLMClient
    from .src.vlm_client import VLMClient
except ImportError:
    from src.llm_client import LLMClient
    from src.vlm_client import VLMClient


# Provider flag -> (client provider prefix, LLM model, VLM model)
MODELS = {
    "openai": ("openai", "gpt-5-mini", "gpt-5-mini"),
    "anthropic": ("anthropic", "claude-opus-5.5", "claude-opus-5.5"),
    "groq": ("groq", "openai-oss-120b", "qwen3.6-27b"),
    "nebius": ("nebius", "kimi-k2.6", "qwen3-2.5-70b"),
    "google": ("gemini", "gemini-1.5-flash", "gemini-1.5-flash"),
}

COMMON_PARAMETERS = {
    "max_tokens": 4096,
}

PROVIDER_PARAMETERS = {
    "anthropic": {"effort": "low"},
    # Lower temperature and keep Qwen3 reasoning out of the content, so strict JSON schemas validate.
    "groq": {"temperature": 0.2, "reasoning_format": "parsed"},
}

TEST_IMAGES_DIR = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(__file__))), "test_images")
PEOPLE_IMAGE_PATH = os.path.join(TEST_IMAGES_DIR, "test_silvio.jpg")
AFFORDANCES_IMAGE_PATH = os.path.join(TEST_IMAGES_DIR, "robotic_affordances.jpg")


class SkipTest(Exception):
    pass


# ---------------------------------------------------------------- output schemas

class KeyEntity(BaseModel):
    entity: str
    type: str


class EmailClassification(BaseModel):
    category: str
    priority: str
    confidence_score: float
    sentiment: str
    key_entities: list[KeyEntity]
    suggested_actions: list[str]
    requires_immediate_attention: bool
    estimated_response_time: str


class SceneDescription(BaseModel):
    people_count: int
    setting: str
    objects: list[str]
    main_colors: list[str]


class BoundingBox(BaseModel):
    x_min: int = Field(description="Top-left x, in pixels")
    y_min: int = Field(description="Top-left y, in pixels")
    x_max: int = Field(description="Bottom-right x, in pixels")
    y_max: int = Field(description="Bottom-right y, in pixels")


class LabeledBoundingBox(BoundingBox):
    label: str


class BoundingBoxes(BaseModel):
    bounding_boxes: list[LabeledBoundingBox]


class Point(BaseModel):
    x: int = Field(description="x, in pixels")
    y: int = Field(description="y, in pixels")


class GraspAffordance(BaseModel):
    object_label: str
    bounding_box: BoundingBox
    grasp_point: Point  # where the center of the gripper should close; described in the prompt, since a description next to a $ref is rejected by some strict validators
    gripper_angle_deg: float = Field(description="In-plane gripper angle in degrees, in [-90, 90]: 0 means the fingers close along the image x axis, positive is counter-clockwise")
    grasp_type: str = Field(description="One of: pinch, power, handle, edge")
    confidence: float = Field(description="Confidence in [0, 1]")


class GraspAffordances(BaseModel):
    affordances: list[GraspAffordance]


# ---------------------------------------------------------------- helpers

def build_client(client_class, provider: str):
    prefix, llm_model, vlm_model = MODELS[provider]
    model = llm_model if client_class is LLMClient else vlm_model
    return client_class(
        model_name=f"{prefix}/{model}",
        **COMMON_PARAMETERS,
        **PROVIDER_PARAMETERS.get(provider, {}),
    )


def image_size(image_path: str):
    if not os.path.isfile(image_path):
        raise SkipTest(f"image not found: {image_path}")
    with Image.open(image_path) as image:
        return image.size


def check_box_inside(box: BoundingBox, width: int, height: int, name: str):
    assert 0 <= box.x_min < box.x_max <= width and 0 <= box.y_min < box.y_max <= height, \
        f"{name}: box {box.model_dump()} is not a valid box inside the {width}x{height} image"


def draw_annotations(image_path: str, boxes: list, points: list = (), title: str = ""):
    """boxes: [(label, BoundingBox)], points: [(Point, angle_deg)]."""
    import math
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for label, box in boxes:
        draw.rectangle([box.x_min, box.y_min, box.x_max, box.y_max], outline="red", width=3)
        draw.text((box.x_min, max(0, box.y_min - 12)), label, fill="red")
    for point, angle_deg in points:
        # Gripper closing direction drawn as a line through the grasp point.
        dx = 25 * math.cos(math.radians(angle_deg))
        dy = -25 * math.sin(math.radians(angle_deg))
        draw.line([point.x - dx, point.y - dy, point.x + dx, point.y + dy], fill="lime", width=3)
        draw.ellipse([point.x - 5, point.y - 5, point.x + 5, point.y + 5], fill="lime")
    image.show(title=title)


# ---------------------------------------------------------------- LLM tests

def llm_open_question(provider: str, show: bool):
    llm = build_client(LLMClient, provider)
    response = llm(
        system_message="You are a helpful assistant, with strong critical and analytical skills, but very smart at emotional intelligence.",
        user_message="What would be a good travel itinerary to move from Italian capital to italian finance capital?",
    )
    assert isinstance(response, str) and response.strip(), "empty response"
    print(response)
    return llm


def llm_structured_output(provider: str, show: bool):
    llm = build_client(LLMClient, provider)
    response = llm(
        system_message="You are an email classification expert. Classify emails into structured categories with confidence scores, priority levels, and suggested actions.",
        user_message="Subject: URGENT: Server downtime affecting production\n\nHi Team,\n\nOur main production server went down at 2:30 PM EST. Customer-facing services are currently unavailable. We need immediate action to restore services. Please join the emergency call.\n\nBest regards,\nDevOps Team",
        force_json=True,
        forced_json_schema=EmailClassification,
    )
    email_classification = EmailClassification.model_validate(json.loads(response))
    print(json.dumps(email_classification.model_dump(), indent=2))
    return llm


# ---------------------------------------------------------------- VLM tests

def vlm_image_question(provider: str, show: bool):
    image_size(PEOPLE_IMAGE_PATH)
    vlm = build_client(VLMClient, provider)
    response = vlm(text_prompt="Describe this image in a few sentences.", image=PEOPLE_IMAGE_PATH)
    assert isinstance(response, str) and response.strip(), "empty response"
    print(response)
    return vlm


def vlm_image_structured_output(provider: str, show: bool):
    image_size(PEOPLE_IMAGE_PATH)
    vlm = build_client(VLMClient, provider)
    response = vlm(
        text_prompt="Describe the scene in this image: how many people are visible, the setting, the visible objects and the main colors.",
        image=PEOPLE_IMAGE_PATH,
        force_json=True,
        forced_json_schema=SceneDescription,
    )
    scene = SceneDescription.model_validate(json.loads(response))
    assert scene.people_count >= 0, f"negative people_count: {scene.people_count}"
    print(json.dumps(scene.model_dump(), indent=2))
    return vlm


def vlm_bounding_boxes(provider: str, show: bool):
    width, height = image_size(PEOPLE_IMAGE_PATH)
    vlm = build_client(VLMClient, provider)
    response = vlm(
        text_prompt=(
            "Find all the faces in the image. If there are specific known people, label them with their names. "
            f"The image is {width} x {height} pixels; return the pixel coordinates of each bounding box."
        ),
        image=PEOPLE_IMAGE_PATH,
        force_json=True,
        forced_json_schema=BoundingBoxes,
    )
    result = BoundingBoxes.model_validate(json.loads(response))
    print(json.dumps(result.model_dump(), indent=2))
    for box in result.bounding_boxes:
        check_box_inside(box, width, height, box.label)
    if show:
        draw_annotations(PEOPLE_IMAGE_PATH, [(box.label, box) for box in result.bounding_boxes], title=f"{provider} faces")
    return vlm


def vlm_grasp_affordances(provider: str, show: bool):
    width, height = image_size(AFFORDANCES_IMAGE_PATH)
    vlm = build_client(VLMClient, provider)
    response = vlm(
        text_prompt=(
            "You are the perception module of a robot arm equipped with a parallel-jaw gripper. "
            "List every object in the image that the robot could grasp. For each object give its bounding box, "
            "the grasp point where the center of the gripper should close, the in-plane gripper angle, "
            "the grasp type and your confidence. "
            f"The image is {width} x {height} pixels; all coordinates are in pixels."
        ),
        image=AFFORDANCES_IMAGE_PATH,
        force_json=True,
        forced_json_schema=GraspAffordances,
    )
    result = GraspAffordances.model_validate(json.loads(response))
    print(json.dumps(result.model_dump(), indent=2))
    for affordance in result.affordances:
        check_box_inside(affordance.bounding_box, width, height, affordance.object_label)
        point = affordance.grasp_point
        assert 0 <= point.x <= width and 0 <= point.y <= height, \
            f"{affordance.object_label}: grasp point {point.model_dump()} is outside the {width}x{height} image"
        assert -90 <= affordance.gripper_angle_deg <= 90, \
            f"{affordance.object_label}: gripper angle {affordance.gripper_angle_deg} outside [-90, 90]"
    if show:
        draw_annotations(
            AFFORDANCES_IMAGE_PATH,
            [(a.object_label, a.bounding_box) for a in result.affordances],
            [(a.grasp_point, a.gripper_angle_deg) for a in result.affordances],
            title=f"{provider} grasp affordances",
        )
    return vlm


LLM_TESTS = [llm_open_question, llm_structured_output]
VLM_TESTS = [vlm_image_question, vlm_image_structured_output, vlm_bounding_boxes, vlm_grasp_affordances]


# ---------------------------------------------------------------- runner

def run(providers: list, suites: list, show: bool) -> bool:
    tests = (LLM_TESTS if "llm" in suites else []) + (VLM_TESTS if "vlm" in suites else [])
    results = []
    for provider in providers:
        prefix = MODELS[provider][0]
        api_key_name = f"{prefix.upper()}_API_KEY"
        for test in tests:
            print(f"\n### [{provider}] {test.__name__}")
            if not os.getenv(api_key_name):
                results.append((provider, test.__name__, "SKIP", f"{api_key_name} not set", None, None))
                print(f"SKIP: {api_key_name} not set")
                continue
            start = time.perf_counter()
            try:
                client = test(provider, show)
                elapsed = time.perf_counter() - start
                usage = client.get_total_usage()["total_tokens"]
                results.append((provider, test.__name__, "PASS", "", elapsed, usage))
                print(f"-> {elapsed:.1f}s, {usage} tokens")
            except (SkipTest, NotImplementedError) as e:
                results.append((provider, test.__name__, "SKIP", str(e), None, None))
                print(f"SKIP: {e}")
            except Exception as e:
                print(format_exc())
                results.append((provider, test.__name__, "FAIL", f"{type(e).__name__}: {e}"[:80], time.perf_counter() - start, None))

    print("\n=== Summary ===")
    for provider, name, status, reason, elapsed, usage in results:
        stats = f"{elapsed:6.1f}s {usage if usage is not None else '-':>7} tok" if elapsed is not None else " " * 18
        print(f"{status:4}  {provider:9}  {name:28} {stats}  {reason}")
    return all(status != "FAIL" for _, _, status, *_ in results)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LLM / VLM examples on the selected providers.")
    parser.add_argument("--llm", action="store_true", help="run the LLM suite")
    parser.add_argument("--vlm", action="store_true", help="run the VLM suite")
    for provider in MODELS:
        parser.add_argument(f"--{provider}", action="store_true", help=f"run on {provider}")
    parser.add_argument("--show", action="store_true", help="open annotated images for bounding boxes and affordances")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suites = [suite for suite in ("llm", "vlm") if getattr(args, suite)] or ["llm", "vlm"]
    providers = [provider for provider in MODELS if getattr(args, provider)] or list(MODELS)
    sys.exit(0 if run(providers, suites, args.show) else 1)
