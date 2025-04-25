import pytest
from unittest.mock import patch, MagicMock
import requests # Import requests to mock its methods

# Import the function to test
from mcp_server.llm_interaction import get_root_cause_hint

# --- Test Setup ---

@pytest.fixture
def ollama_config():
    """Fixture providing default Ollama config for tests."""
    return {
        "base_url": "http://localhost:11434",
        "model": "test-model",
        "request_timeout": 5,
    }

# --- Test Cases ---

@patch('requests.post') # Mock the requests.post call
def test_get_root_cause_hint_success(mock_post, ollama_config):
    """Test successful interaction with Ollama API."""
    context_str = "Host: server1\nLog: Error line\n"
    test_report = "Test timed out"

    # Configure the mock response from requests.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": ollama_config['model'],
        "created_at": "2023-10-26T18:23:45.123Z",
        "message": {
            "role": "assistant",
            "content": "  Based on the log, check the 'Error line'.  " # Include spaces to test strip()
        },
        "done": True
        # Add other fields Ollama might return
    }
    # Make requests.post return our mock response
    mock_post.return_value = mock_response

    hint = get_root_cause_hint(context_str, test_report, ollama_config)

    # Assertions
    assert hint == "Based on the log, check the 'Error line'." # Check that strip() worked

    # Verify requests.post was called correctly
    expected_url = f"{ollama_config['base_url']}/api/chat"
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == expected_url # Check URL
    assert call_kwargs['timeout'] == ollama_config['request_timeout'] # Check timeout

    # Check payload structure (important parts)
    payload = call_kwargs['json']
    assert payload['model'] == ollama_config['model']
    assert payload['stream'] is False
    assert len(payload['messages']) == 1
    assert payload['messages'][0]['role'] == 'user'
    assert test_report in payload['messages'][0]['content']
    assert context_str in payload['messages'][0]['content']

@patch('requests.post')
def test_get_root_cause_hint_request_exception(mock_post, ollama_config):
    """Test handling of network errors when calling Ollama."""
    context_str = "Some context"
    test_report = "Some failure"

    # Configure requests.post to raise a connection error
    error_message = "Connection refused"
    mock_post.side_effect = requests.exceptions.RequestException(error_message)

    hint = get_root_cause_hint(context_str, test_report, ollama_config)

    # Assertions
    assert "Connection error with Ollama: Connection refused" in hint
    assert error_message in hint
    mock_post.assert_called_once()

@patch('requests.post')
def test_get_root_cause_hint_timeout(mock_post, ollama_config):
    """Test handling of timeouts when calling Ollama."""
    context_str = "Some context"
    test_report = "Some failure"

    # Configure requests.post to raise a timeout error
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

    hint = get_root_cause_hint(context_str, test_report, ollama_config)

    # Assertions
    assert "Timeout while contacting Ollama" in hint
    assert f"(limit: {ollama_config['request_timeout']}s)" in hint
    mock_post.assert_called_once()

@patch('requests.post')
def test_get_root_cause_hint_http_error(mock_post, ollama_config):
    """Test handling of HTTP error responses from Ollama (e.g., 4xx, 5xx)."""
    context_str = "Some context"
    test_report = "Some failure"

    # Configure mock response with an error status code
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.reason = "Internal Server Error"
    # Configure raise_for_status to simulate the error being raised by requests
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Server Error: Internal Server Error for url", response=mock_response
    )
    mock_post.return_value = mock_response

    hint = get_root_cause_hint(context_str, test_report, ollama_config)

    # Assertions
    # The RequestException handler should catch this
    assert "Internal Server Error for url" in hint
    assert "500 Server Error" in hint # Check that the HTTPError message is included
    mock_post.assert_called_once()

def test_get_root_cause_hint_missing_config(ollama_config):
    """Test behavior when Ollama config is missing or incomplete."""
    context_str = "Some context"
    test_report = "Some failure"

    hint_none = get_root_cause_hint(context_str, test_report, None)
    assert "Error: Ollama configuration is not available." in hint_none

    hint_no_url = get_root_cause_hint(context_str, test_report, {"model": "test"}) # Missing base_url
    assert "Error: Ollama configuration is not available." in hint_no_url

# Add tests for different prompt variations if needed.
