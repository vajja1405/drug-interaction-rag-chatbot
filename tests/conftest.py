"""
tests/conftest.py
─────────────────
Pytest fixtures and configuration to ensure tests run fast and offline.
Mocks the OpenAI LLM to prevent external network calls during testing.
"""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
import os

# Set dummy key for tests before importing the app
os.environ["OPENAI_API_KEY"] = "sk-test-dummy"

from config import settings
from api.server import app

@pytest.fixture(autouse=True)
def mock_openai_llm():
    """Globally mocks ChatOpenAI to return a deterministic JSON blob."""
    mock_response = MagicMock()
    mock_response.content = '''```json
{
  "pairs": [
    {
      "drug_a": "warfarin",
      "drug_b": "ibuprofen",
      "mechanism": "Mocked mechanism from LLM.",
      "clinical_effects": "Mocked clinical effects.",
      "management": "Mocked management instructions.",
      "monitoring_points": ["Monitor A", "Monitor B"],
      "interaction_type": "Pharmacodynamic",
      "evidence_quality": "Moderate"
    }
  ],
  "overall_summary": "Mocked narrative summary.",
  "monitoring_priorities": ["Monitor A", "Monitor B"]
}
```'''
    
    with patch("chatbot.interaction_agent.ChatOpenAI") as mock_chat:
        # The agent instantiates ChatOpenAI, then .invoke() is called
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_instance.stream.return_value = iter([mock_response])
        mock_chat.return_value = mock_instance
        yield mock_chat

@pytest.fixture
def client():
    """FastAPI TestClient for integration tests without networking."""
    with TestClient(app) as test_client:
        yield test_client
