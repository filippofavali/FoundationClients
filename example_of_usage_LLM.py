"""
Example of usage of the LLM client. This file is meant to be run as a script
"""

try:
    from .src.llm_client import LLMClient
except ImportError:
    from src.llm_client import LLMClient
    

def test_groq():
    params = {
        "model_name": "groq/openai-oss-120b",
        'temperature': 0.7,
        'max_tokens': 2048,
        'top_p': 0.9
    }
    llm_client = LLMClient(
        **params
    )
    system_message = "You are a helpful assistant, with strong critical and analytical skills, but very smart at emotional intelligence."
    user_message = "What would be a good travel itinerary to move from Italian capital to italian finance capital?"
    response = llm_client(
        system_message=system_message,
        user_message=user_message
    )
    print(response)


def test_groq_strict_mode():
    from pydantic import BaseModel
    import json
    params = {
        "model_name": "groq/openai-oss-120b",
        'temperature': 0.7,
        'max_tokens': 2048,
        'top_p': 0.9
    }
    llm_client = LLMClient(
        **params
    )
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

    system_message="You are an email classification expert. Classify emails into structured categories with confidence scores, priority levels, and suggested actions."
    user_message="Subject: URGENT: Server downtime affecting production\\n\\nHi Team,\\n\\nOur main production server went down at 2:30 PM EST. Customer-facing services are currently unavailable. We need immediate action to restore services. Please join the emergency call.\\n\\nBest regards,\\nDevOps Team"
    response = llm_client(
        system_message=system_message,
        user_message=user_message,
        force_json=True,
        forced_json_schema=EmailClassification
    )
    
    email_classification = EmailClassification.model_validate(json.loads(response))
    print(json.dumps(email_classification.model_dump(), indent=2))


def test_nebius():
    return {
        "model_name": "nebius/qwen3-2.5-70b",
        'temperature': 0.7,
        'max_tokens': 2048,
        'top_p': 0.9,
        'logprobs': False,
        'full_content': False
    }

if __name__ == "__main__":
    
    use_nebius = False
    use_groq = True

    if use_nebius:
        test_nebius()
    elif use_groq:
        test_groq()
        test_groq_strict_mode()


