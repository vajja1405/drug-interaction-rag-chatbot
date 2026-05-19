"""
tests/test_api_integration.py
─────────────────────────────
End-to-end FastAPI test using the test client.
Uses the globally mocked LLM to run offline and fast.
"""
import pytest
from fastapi.testclient import TestClient

def test_health_endpoint(client: TestClient):
    """Ensure the API boots and responds to health checks."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_analyze_interaction_endpoint(client: TestClient):
    """End-to-end test of the interaction analysis pipeline."""
    payload = {
        "drugs": ["warfarin", "ibuprofen"]
    }
    
    response = client.post("/analyze_interaction", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify the overall structure
    assert "drugs" in data
    assert "interactions" in data
    
    # Check that the pair was analyzed
    assert len(data["interactions"]) >= 1
    interaction = data["interactions"][0]
    assert "ibuprofen" in interaction["pair"]
    assert "warfarin" in interaction["pair"]
    assert interaction["severity"] in ["Severe", "Moderate", "Low", "Unknown"]
    
    # The explanation fields should match our mocked LLM output
    assert "mechanism" in interaction
