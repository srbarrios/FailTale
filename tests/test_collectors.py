import pytest
from unittest.mock import MagicMock, call

# Import the base class and specific collectors
from mcp_server_poc.collectors.base_collector import BaseCollector
from mcp_server_poc.collectors.server_collector import ServerCollector
# from mcp_server_poc.collectors.minion_collector import MinionCollector # etc.

# --- Test Setup ---

# Mock the SSH executor function
@pytest.fixture
def mock_ssh_executor():
    """Fixture to provide a mock SSH executor function."""
    # This mock will simulate the behavior of execute_remote_command
    # You can configure its return values per test case
    return MagicMock()

@pytest.fixture
def default_ssh_config():
    """Default SSH config for tests."""
    return {
        "username": "testuser",
        "private_key_path": "/fake/key.pem",
        "port": 22,
        "connection_timeout": 5,
        "command_timeout": 15,
    }

# --- Test Cases for BaseCollector (if needed, though it's abstract) ---

def test_base_collector_init(mock_ssh_executor, default_ssh_config):
    """Test BaseCollector initialization (indirectly via a subclass)."""
    hostname = "basehost"
    # Need a concrete subclass to instantiate
    class DummyCollector(BaseCollector):
        def collect(self, data_definitions):
            return [] # Dummy implementation

    collector = DummyCollector(hostname, mock_ssh_executor, default_ssh_config)
    assert collector.hostname == hostname
    assert collector.ssh_executor == mock_ssh_executor
    assert collector.ssh_config == default_ssh_config

def test_base_collector_init_missing_params(mock_ssh_executor, default_ssh_config):
    """Test BaseCollector init raises error if params are missing."""
    class DummyCollector(BaseCollector):
        def collect(self, data_definitions): return []
    with pytest.raises(ValueError):
        DummyCollector(None, mock_ssh_executor, default_ssh_config)
    with pytest.raises(ValueError):
        DummyCollector("host", None, default_ssh_config)
    with pytest.raises(ValueError):
        DummyCollector("host", mock_ssh_executor, None)

# --- Test Cases for ServerCollector ---

def test_server_collector_collect_success(mock_ssh_executor, default_ssh_config):
    """Test ServerCollector successfully collects data."""
    hostname = "server1"
    ssh_config = default_ssh_config
    data_defs = [
        {"description": "Get logs", "command": "tail /var/log/server.log"},
        {"description": "Check status", "command": "systemctl status server"},
    ]

    # Configure the mock executor's return values for the expected calls
    mock_ssh_executor.side_effect = [
        ("Log line 1\nLog line 2", "", 0), # Result for first command
        ("Active: active (running)", "", 0), # Result for second command
    ]

    collector = ServerCollector(hostname, mock_ssh_executor, ssh_config)
    results = collector.collect(data_defs)

    # Assertions
    assert len(results) == 2

    # Check first result
    assert results[0]['description'] == "Get logs"
    assert results[0]['command'] == "tail /var/log/server.log"
    assert results[0]['status'] == 0
    assert results[0]['output'] == "Log line 1\nLog line 2"
    assert results[0]['error'] == ""

    # Check second result
    assert results[1]['description'] == "Check status"
    assert results[1]['command'] == "systemctl status server"
    assert results[1]['status'] == 0
    assert results[1]['output'] == "Active: active (running)"
    assert results[1]['error'] == ""

    # Verify the mock executor was called correctly
    expected_calls = [
        call(
            hostname=hostname, username=ssh_config['username'],
            private_key_path=ssh_config['private_key_path'], command=data_defs[0]['command'],
            port=ssh_config['port'], conn_timeout=ssh_config['connection_timeout'],
            cmd_timeout=ssh_config['command_timeout']
        ),
        call(
            hostname=hostname, username=ssh_config['username'],
            private_key_path=ssh_config['private_key_path'], command=data_defs[1]['command'],
            port=ssh_config['port'], conn_timeout=ssh_config['connection_timeout'],
            cmd_timeout=ssh_config['command_timeout']
        ),
    ]
    mock_ssh_executor.assert_has_calls(expected_calls)
    assert mock_ssh_executor.call_count == 2

def test_server_collector_collect_command_failure(mock_ssh_executor, default_ssh_config):
    """Test ServerCollector handles a command failing."""
    hostname = "server2"
    ssh_config = default_ssh_config
    data_defs = [
        {"description": "Run failing command", "command": "false"},
    ]

    # Configure mock for failure
    mock_ssh_executor.return_value = (None, "Command failed", 1)

    collector = ServerCollector(hostname, mock_ssh_executor, ssh_config)
    results = collector.collect(data_defs)

    # Assertions
    assert len(results) == 1
    assert results[0]['status'] == 1
    assert results[0]['output'] is None
    assert results[0]['error'] == "Command failed"
    mock_ssh_executor.assert_called_once()

def test_server_collector_collect_missing_command(mock_ssh_executor, default_ssh_config):
    """Test ServerCollector handles a data definition with no command."""
    hostname = "server3"
    ssh_config = default_ssh_config
    data_defs = [
        {"description": "No command here"}, # Missing 'command' key
    ]

    collector = ServerCollector(hostname, mock_ssh_executor, ssh_config)
    results = collector.collect(data_defs)

    # Assertions
    assert len(results) == 1
    assert results[0]['description'] == "No command here"
    assert results[0]['command'] is None
    assert results[0]['status'] == -100 # Or your chosen code for config errors
    assert results[0]['error'] == "Command missing in config"
    mock_ssh_executor.assert_not_called() # SSH should not be attempted

def test_server_collector_no_data_definitions(mock_ssh_executor, default_ssh_config):
    """Test ServerCollector with an empty list of data definitions."""
    hostname = "server4"
    ssh_config = default_ssh_config
    data_defs = [] # Empty list

    collector = ServerCollector(hostname, mock_ssh_executor, ssh_config)
    results = collector.collect(data_defs)

    # Assertions
    assert len(results) == 0
    mock_ssh_executor.assert_not_called()

# Add tests for other collectors (MinionCollector, ProxyCollector) following the same pattern.
# Add tests for edge cases like missing SSH config in the dictionary.
