import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies if not installed
try:
    import pandas as pd
except ImportError:
    # Create a mock pandas module
    mock_pd = MagicMock()
    mock_pd.DataFrame = MagicMock
    mock_pd.Timestamp.now.return_value = "2024-01-01"
    sys.modules["pandas"] = mock_pd
    import pandas as pd

try:
    import dotenv
except ImportError:
    mock_dotenv = MagicMock()
    sys.modules["dotenv"] = mock_dotenv

# Mock generic google module for direct imports in clients
try:
    import google.genai
except ImportError:
    mock_google = MagicMock()
    mock_genai_pkg = MagicMock()
    mock_google.genai = mock_genai_pkg
    sys.modules["google"] = mock_google
    sys.modules["google.genai"] = mock_genai_pkg

# Now import the module to test
from . import base_client
from .llm_client import LLMClient
from .vlm_client import VLMClient
from .base_client import ModelRegistry

class TestFoundationClients(unittest.TestCase):

    def setUp(self):
        # Patch the SDK imports in the module
        self.groq_patcher = patch('src.base_client.Groq')
        self.openai_patcher = patch('src.base_client.OpenAI')
        self.anthropic_patcher = patch('src.base_client.Anthropic')
        self.genai_patcher = patch('src.base_client.genai')

        self.mock_groq = self.groq_patcher.start()
        self.mock_openai = self.openai_patcher.start()
        self.mock_anthropic = self.anthropic_patcher.start()
        self.mock_genai = self.genai_patcher.start()
        
        # Setup mock return values
        self.mock_groq_instance = self.mock_groq.return_value
        self.mock_openai_instance = self.mock_openai.return_value
        self.mock_anthropic_instance = self.mock_anthropic.return_value
        # Gemini
        self.mock_genai_client = self.mock_genai.Client.return_value

        # Ensure clients are initialized with mocks even if imports failed in real module
        base_client.Groq = self.mock_groq
        base_client.OpenAI = self.mock_openai
        base_client.Anthropic = self.mock_anthropic
        base_client.genai = self.mock_genai

    def tearDown(self):
        self.groq_patcher.stop()
        self.openai_patcher.stop()
        self.anthropic_patcher.stop()
        self.genai_patcher.stop()

    def test_registry(self):
        self.assertEqual(ModelRegistry.get_model_id("groq", "llama3.1-8b"), "llama-3.1-8b-instant")
        self.assertEqual(ModelRegistry.get_model_id("openai", "gpt-4o"), "gpt-4o")
        self.assertEqual(ModelRegistry.get_model_id("unknown", "test-model"), "test-model")

    def test_llm_client_groq_init(self):
        client = LLMClient(model_name="groq/llama3.1-8b", api_key="test_key")
        self.assertEqual(client.provider, "groq")
        self.assertEqual(client.model_name, "llama-3.1-8b-instant")
        self.mock_groq.assert_called_with(api_key="test_key")

    def test_llm_client_call_groq(self):
        client = LLMClient(model_name="groq/llama3.1-8b", api_key="test_key")
        
        # Mock response
        mock_chat = self.mock_groq_instance.chat.completions.create
        mock_message = MagicMock()
        mock_message.choices[0].message.content = "Test response"
        mock_message.usage.prompt_tokens = 10
        mock_message.usage.completion_tokens = 20
        mock_chat.return_value = mock_message

        response = client("Hello")
        self.assertEqual(response, "Test response")
        
        # Check metrics (assuming pandas fits or is mocked)
        if hasattr(client, 'usage_metrics') and client.usage_metrics is not None:
             # If using real pandas or mock behaving like it
             pass

    def test_vlm_client_openai_call(self):
        client = VLMClient(model_name="openai/gpt-4o", api_key="test_key")
        
        mock_create = self.mock_openai_instance.chat.completions.create
        mock_message = MagicMock()
        mock_message.choices[0].message.content = "Image description"
        mock_message.usage.prompt_tokens = 50
        mock_message.usage.completion_tokens = 10
        mock_create.return_value = mock_message

        # Mock opening file/image or pass url
        response = client("Describe", "https://example.com/image.jpg")
        self.assertEqual(response, "Image description")
        
        # Verify call arguments structure
        call_args = mock_create.call_args[1]
        self.assertEqual(call_args['model'], "gpt-4o")
        # Structure is messages -> content -> [text, image_url]
        self.assertEqual(call_args['messages'][0]['content'][1]['type'], "image_url")

if __name__ == '__main__':
    unittest.main()
