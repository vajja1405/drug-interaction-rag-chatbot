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

MOCK_LLM_JSON = '''```json
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


@pytest.fixture(autouse=True)
def mock_openai_llm():
    """Replaces ChatOpenAI with LangChain's FakeListChatModel returning a deterministic JSON blob.

    A real Runnable test double (rather than MagicMock) keeps `prompt | llm | parser` chains
    working on current langchain-core, which LangGraph requires (>= 0.3.23)."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    content = MOCK_LLM_JSON
    with patch("chatbot.interaction_agent.ChatOpenAI") as mock_chat:
        mock_chat.side_effect = lambda *a, **k: FakeListChatModel(responses=[content] * 50)
        yield mock_chat


@pytest.fixture
def client():
    """FastAPI TestClient for integration tests without networking."""
    with TestClient(app) as test_client:
        yield test_client
