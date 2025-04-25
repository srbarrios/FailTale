import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock, ANY, call # Import call
import asyncio

# Assuming your Flask app instance is in mcp_server.server
# Adjust the import path if your structure is different
from mcp_server.server import app, load_environments # Import app and the loader function

# --- Test Setup ---

# Define a sample config for testing
SAMPLE_CONFIG = {
    "ssh_defaults": {
        "username": "testuser",
        "private_key_path": "/fake/id_rsa",
        "port": 22,
        "connection_timeout": 10,
        "command_timeout": 30,
        "passphrase": None,
        "password": None,
    },
    "components": {
        "server": {
            "useful_data": [
                {"description": "Server Log", "command": "cat /log/server.log"},
                {"description": "Server Status", "command": "systemctl status server"},
            ]
        },
        "proxy": {
            "useful_data": [
                {"description": "Proxy Log", "command": "cat /log/proxy.log"},
            ]
        }
    },
    "ollama": {
        "base_url": "http://mock-ollama:11434",
        "model": "test-model",
        "request_timeout": 5,
    }
}

@pytest.fixture(scope="function", autouse=True)
def setup_config():
    """Fixture to load a known config before each test function."""
    # Use patch to temporarily replace the global 'config' variable
    # in the server module where it's looked up.
    with patch('mcp_server.server.config', SAMPLE_CONFIG):
        yield

@pytest.fixture
def client():
    """Pytest fixture for the Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- Test Cases ---

def test_index_route(client):
    """Test the index route."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['message'] == "MCP Server PoC is running!"

# --- Tests for /mcp/v1/collect Endpoint ---

# --- FIX: Correct patch target for asyncio.run ---
@patch('asyncio.run') # Patch the standard library asyncio.run directly
# Keep mocking the ssh executor as it's still part of the logic flow
@patch('mcp_server.server.execute_remote_command_async', new_callable=AsyncMock)
def test_collect_endpoint_success(mock_ssh_execute, mock_asyncio_run, client):
    """Test the /collect endpoint mocking asyncio.run."""
    # Define the expected final dictionary structure that the endpoint should return
    # This is what asyncio.run(run_all()) would return if run_all worked correctly.
    expected_final_dict = {
        "server1.test": {
            "role": "server",
            "data": [
                {"description": "Server Log", "command": "cat /log/server.log", "status": 0, "output": "Server log line 1", "error": ""},
                {"description": "Server Status", "command": "systemctl status server", "status": 0, "output": "Active: active", "error": ""},
            ]
        },
        "proxy1.test": {
            "role": "proxy",
            "data": [
                {"description": "Proxy Log", "command": "cat /log/proxy.log", "status": 0, "output": "Proxy access line", "error": ""},
            ]
        }
    }
    # Configure the mock for asyncio.run to return this pre-calculated result directly.
    mock_asyncio_run.return_value = expected_final_dict

    # Configure the side effect for the underlying SSH executor, even though
    # it won't be directly awaited when asyncio.run is mocked. This ensures
    # the mock is set up correctly if the patching changes later.
    mock_ssh_execute.side_effect = [
        ("Server log line 1", "", 0), # server1, command 1
        ("Active: active", "", 0),    # server1, command 2
        ("Proxy access line", "", 0), # proxy1, command 1
    ]

    request_payload = {
        "hosts": [
            {"hostname": "server1.test", "role": "server"},
            {"hostname": "proxy1.test", "role": "proxy"}
        ]
    }

    response = client.post('/mcp/v1/collect', json=request_payload)
    data = json.loads(response.data)

    # Assertions
    assert response.status_code == 200
    # Check that the data returned by the endpoint matches the mocked asyncio.run result
    assert data == expected_final_dict

    # Verify asyncio.run was called once.
    mock_asyncio_run.assert_called_once()
    # The first argument passed to asyncio.run should be the 'run_all' coroutine object
    assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])


# --- FIX: Correct patch target for asyncio.run ---
@patch('asyncio.run') # Patch the standard library asyncio.run directly
@patch('mcp_server.server.execute_remote_command_async', new_callable=AsyncMock)
def test_collect_endpoint_partial_failure(mock_ssh_execute, mock_asyncio_run, client):
    """Test /collect mocking asyncio.run when one command fails."""
    # Define the expected final dictionary structure
    expected_final_dict = {
        "server1.test": {
            "role": "server",
            "data": [
                {"description": "Server Log", "command": "cat /log/server.log", "status": 0, "output": "Server log ok", "error": ""},
                {"description": "Server Status", "command": "systemctl status server", "status": -5, "output": None, "error": "Connection error after retries: Connection timed out"},
            ]
        },
        "proxy1.test": {
            "role": "proxy",
            "data": [
                {"description": "Proxy Log", "command": "cat /log/proxy.log", "status": 0, "output": "Proxy log ok", "error": ""},
            ]
        }
    }
    # Configure the mock for asyncio.run to return this result
    mock_asyncio_run.return_value = expected_final_dict

    # Configure side effects for the underlying (but effectively bypassed) SSH calls
    mock_ssh_execute.side_effect = [
        ("Server log ok", "", 0), # server1, command 1 (success)
        (None, "Connection error after retries: Connection timed out", -5), # server1, command 2 (fail)
        ("Proxy log ok", "", 0), # proxy1, command 1 (success)
    ]

    request_payload = {
        "hosts": [
            {"hostname": "server1.test", "role": "server"},
            {"hostname": "proxy1.test", "role": "proxy"}
        ]
    }
    response = client.post('/mcp/v1/collect', json=request_payload)
    data = json.loads(response.data)

    assert response.status_code == 200
    assert data == expected_final_dict
    mock_asyncio_run.assert_called_once()
    assert asyncio.iscoroutine(mock_asyncio_run.call_args[0][0])


def test_collect_endpoint_bad_request_no_json(client):
    """Test /collect with non-JSON data."""
    response = client.post('/mcp/v1/collect', data="not json")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Expected JSON request" in data["error"]

def test_collect_endpoint_bad_request_missing_hosts(client):
    """Test /collect with missing 'hosts' field."""
    response = client.post('/mcp/v1/collect', json={"other_field": "value"})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "hosts" in data["error"]

@patch('mcp_server.server.config', None) # Simulate config not loaded
def test_collect_endpoint_no_config(client):
    """Test /collect when server configuration is not loaded."""
    request_payload = { "hosts": [{"hostname": "server1.test", "role": "server"}] }
    response = client.post('/mcp/v1/collect', json=request_payload)
    assert response.status_code == 500
    data = json.loads(response.data)
    assert "error" in data
    assert "Server configuration not loaded" in data["error"]

# --- Tests for /mcp/v1/analyze Endpoint ---

@patch('mcp_server.server.get_root_cause_hint') # Patch where it's imported in server.py
def test_analyze_endpoint_success(mock_get_hint, client):
    """Test the /analyze endpoint with valid data."""
    ai_response = "AI suggestion: Check the server logs for errors."
    mock_get_hint.return_value = ai_response

    request_payload = {
        "collected_data": {
            "server1.test": {
                "role": "server",
                "data": [{"description": "Log", "output": "Error found", "status": 0}]
            }
        },
        "test_report": "Test failed due to timeout"
    }

    response = client.post('/mcp/v1/analyze', json=request_payload)
    data = json.loads(response.data)

    assert response.status_code == 200
    assert "root_cause_hint" in data
    assert data["root_cause_hint"] == ai_response

    mock_get_hint.assert_called_once()
    call_args, call_kwargs = mock_get_hint.call_args
    assert "Host: server1.test" in call_args[0]
    assert "Error found" in call_args[0]
    assert call_args[1] == "Test failed due to timeout"
    assert call_args[2] == SAMPLE_CONFIG.get('ollama')


def test_analyze_endpoint_bad_request_no_json(client):
    """Test /analyze with non-JSON data."""
    response = client.post('/mcp/v1/analyze', data="not json")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Expected a JSON request" in data["error"]

def test_analyze_endpoint_bad_request_missing_data(client):
    """Test /analyze with missing 'collected_data'."""
    response = client.post('/mcp/v1/analyze', json={"test_report": "Some failure"})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "collected_data" in data["error"]

@patch('mcp_server.server.config', {"ssh_defaults": {}}) # Simulate missing Ollama config
def test_analyze_endpoint_missing_ollama_config(client):
    """Test /analyze when Ollama config is missing."""
    request_payload = {
        "collected_data": {"host1": {"data": []}},
        "test_report": "Some failure"
    }
    response = client.post('/mcp/v1/analyze', json=request_payload)
    assert response.status_code == 500
    data = json.loads(response.data)
    assert "error" in data
    assert "Ollama configuration is not defined" in data["error"]
