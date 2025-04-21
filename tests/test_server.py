import pytest
import json
from unittest.mock import patch, MagicMock

# Assuming your Flask/FastAPI app instance is in mcp_server_poc.server
from mcp_server_poc.server import app # Use app instance directly for Flask testing client

# --- Test Setup ---

@pytest.fixture
def client():
    """Pytest fixture for the Flask test client."""
    app.config['TESTING'] = True
    # You might need to mock config loading if it happens at import time
    # or ensure a dummy config is available for tests
    with patch('mcp_server_poc.server.load_config') as mock_load_config:
        # Provide a mock config for the server endpoints to use
        mock_load_config.return_value = {
            "ssh_defaults": {
                "username": "testuser",
                "private_key_path": "/fake/key.pem",
                "port": 22,
                "connection_timeout": 10,
                "command_timeout": 30,
            },
            "components": {
                "server": {
                    "useful_data": [
                        {"description": "Server Log", "command": "cat /log/server"},
                    ]
                },
                "minion": {
                     "useful_data": [
                        {"description": "Minion Log", "command": "cat /log/minion"},
                    ]
                }
            },
            "ollama": {
                "base_url": "http://mock-ollama:11434",
                "model": "test-model",
                "request_timeout": 5,
            }
        }
        with app.test_client() as client:
            yield client
    # Clean up after test if needed

# --- Test Cases ---

def test_index_route(client):
    """Test the index route."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['message'] == "MCP Server PoC está corriendo!"

# --- Tests for /mcp/v1/collect Endpoint ---

# Mock the SSH executor used by the endpoint
@patch('mcp_server_poc.server.execute_remote_command')
def test_collect_endpoint_success(mock_executor, client):
    """Test the /collect endpoint with a valid request."""
    # Configure the mock executor to return successful results
    mock_executor.side_effect = [
        ("Server log content", "", 0),  # Result for server1
        ("Minion log content", "", 0),  # Result for minion1
    ]

    request_payload = {
        "hosts": [
            {"hostname": "server1.test", "role": "server"},
            {"hostname": "minion1.test", "role": "minion"}
        ]
    }

    response = client.post('/mcp/v1/collect', json=request_payload)
    data = json.loads(response.data)

    # Assertions
    assert response.status_code == 200
    assert "server1.test" in data
    assert "minion1.test" in data
    assert data["server1.test"]["role"] == "server"
    assert len(data["server1.test"]["data"]) == 1
    assert data["server1.test"]["data"][0]["description"] == "Server Log"
    assert data["server1.test"]["data"][0]["output"] == "Server log content"
    assert data["server1.test"]["data"][0]["status"] == 0

    assert data["minion1.test"]["role"] == "minion"
    assert len(data["minion1.test"]["data"]) == 1
    assert data["minion1.test"]["data"][0]["description"] == "Minion Log"
    assert data["minion1.test"]["data"][0]["output"] == "Minion log content"
    assert data["minion1.test"]["data"][0]["status"] == 0

    assert mock_executor.call_count == 2

@patch('mcp_server_poc.server.execute_remote_command')
def test_collect_endpoint_partial_failure(mock_executor, client):
    """Test /collect when one command fails."""
    # Configure the mock executor for one success, one failure
    mock_executor.side_effect = [
        ("Server log content", "", 0),  # Success for server1
        (None, "Permission denied", 1), # Failure for minion1
    ]

    request_payload = {
        "hosts": [
            {"hostname": "server1.test", "role": "server"},
            {"hostname": "minion1.test", "role": "minion"}
        ]
    }
    response = client.post('/mcp/v1/collect', json=request_payload)
    data = json.loads(response.data)

    assert response.status_code == 200
    # Check server success
    assert data["server1.test"]["data"][0]["status"] == 0
    # Check minion failure
    assert data["minion1.test"]["data"][0]["status"] == 1
    assert data["minion1.test"]["data"][0]["error"] == "Permission denied"
    assert data["minion1.test"]["data"][0]["output"] is None

def test_collect_endpoint_bad_request_no_json(client):
    """Test /collect with non-JSON data."""
    response = client.post('/mcp/v1/collect', data="not json")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "JSON" in data["error"]

def test_collect_endpoint_bad_request_missing_hosts(client):
    """Test /collect with missing 'hosts' field."""
    response = client.post('/mcp/v1/collect', json={"other_field": "value"})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "hosts" in data["error"]

# --- Tests for /mcp/v1/analyze Endpoint ---

# Mock the LLM interaction function used by the endpoint
@patch('mcp_server_poc.server.get_ai_hint')
def test_analyze_endpoint_success(mock_get_hint, client):
    """Test the /analyze endpoint with valid data."""
    mock_get_hint.return_value = "AI suggestion: Check the server logs for errors."

    request_payload = {
        "collected_data": {
            "server1.test": {
                "role": "server",
                "data": [{"description": "Log", "output": "Error found", "status": 0}]
            }
        },
        "failure_summary": "Test failed due to timeout"
    }

    response = client.post('/mcp/v1/analyze', json=request_payload)
    data = json.loads(response.data)

    assert response.status_code == 200
    assert "ai_hint" in data
    assert data["ai_hint"] == "AI suggestion: Check the server logs for errors."
    mock_get_hint.assert_called_once()
    # You could add more detailed assertions on the arguments passed to mock_get_hint

def test_analyze_endpoint_bad_request_missing_data(client):
    """Test /analyze with missing 'collected_data'."""
    response = client.post('/mcp/v1/analyze', json={"failure_summary": "Some failure"})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "collected_data" in data["error"]

# Add more tests:
# - Test cases where config is missing or invalid (if not handled by fixture).
# - Test different combinations of host roles and data definitions.
# - Test error handling within the endpoint logic itself.
# - If using FastAPI, the testing setup will be slightly different (using TestClient).
