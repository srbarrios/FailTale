import pytest
from unittest.mock import patch, MagicMock

# Assuming your executor function is in mcp_server_poc.ssh_executor
from mcp_server_poc.ssh_executor import execute_remote_command

# --- Test Setup ---

# You'll likely need to mock the 'paramiko' library heavily here
# to avoid actual SSH connections during tests.

# Example fixture to provide mock SSH client behavior
@pytest.fixture
def mock_paramiko_ssh_client():
    """Fixture to mock paramiko.SSHClient."""
    # Create mocks for the client, channel, stdin, stdout, stderr
    mock_stdout = MagicMock()
    mock_stdout.channel.recv_exit_status.return_value = 0 # Simulate success
    mock_stdout.read.return_value = b"Command output"

    mock_stderr = MagicMock()
    mock_stderr.read.return_value = b"" # Simulate no error output

    mock_channel = MagicMock()
    mock_channel.recv_exit_status.return_value = 0

    mock_client = MagicMock(spec=paramiko.SSHClient)
    # Configure the mock exec_command to return the mock streams
    mock_client.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)
    # Configure connect to not raise errors
    mock_client.connect.return_value = None
    # Configure close to not raise errors
    mock_client.close.return_value = None

    # Patch paramiko.SSHClient to return our mock client instance
    with patch('paramiko.SSHClient', return_value=mock_client) as patched_client:
        yield patched_client, mock_stdout, mock_stderr # Yield mocks for potential assertions

# Mocking RSAKey loading
@pytest.fixture
def mock_paramiko_key():
    """Fixture to mock paramiko key loading."""
    with patch('paramiko.RSAKey.from_private_key_file') as mock_key:
        mock_key.return_value = MagicMock() # Return a dummy key object
        yield mock_key

# --- Test Cases ---

def test_execute_remote_command_success(mock_paramiko_ssh_client, mock_paramiko_key):
    """Test successful command execution."""
    hostname = "testhost"
    username = "testuser"
    key_path = "/fake/key.pem"
    command = "echo 'hello'"

    mock_client, mock_stdout, _ = mock_paramiko_ssh_client # Get the mock client

    output, error, status = execute_remote_command(hostname, username, key_path, command)

    # Assertions
    mock_client.connect.assert_called_once() # Check connect was called
    mock_client.exec_command.assert_called_once_with(command, timeout=30) # Check command execution
    assert status == 0
    assert output == "Command output"
    assert error == ""
    mock_client.close.assert_called_once() # Check close was called

def test_execute_remote_command_failure_status(mock_paramiko_ssh_client, mock_paramiko_key):
    """Test command execution returning a non-zero exit status."""
    hostname = "testhost"
    username = "testuser"
    key_path = "/fake/key.pem"
    command = "exit 1"

    mock_client, mock_stdout, mock_stderr = mock_paramiko_ssh_client

    # Configure mocks for failure
    mock_stdout.channel.recv_exit_status.return_value = 1
    mock_stdout.read.return_value = b""
    mock_stderr.read.return_value = b"Something went wrong"

    output, error, status = execute_remote_command(hostname, username, key_path, command)

    # Assertions
    assert status == 1
    assert output == ""
    assert error == "Something went wrong"
    mock_client.close.assert_called_once()

def test_execute_remote_command_ssh_exception(mock_paramiko_ssh_client, mock_paramiko_key):
    """Test handling of SSH connection errors."""
    hostname = "unreachable"
    username = "testuser"
    key_path = "/fake/key.pem"
    command = "ls"

    mock_client, _, _ = mock_paramiko_ssh_client
    # Configure the mock connect method to raise an SSHException
    mock_client.connect.side_effect = paramiko.SSHException("Connection failed")

    output, error, status = execute_remote_command(hostname, username, key_path, command)

    # Assertions
    assert status == -1 # Or your chosen error code for SSH errors
    assert output is None
    assert "Error de conexión SSH: Connection failed" in error
    mock_client.exec_command.assert_not_called() # Ensure command wasn't executed
    mock_client.close.assert_called_once() # Close should still be attempted

def test_execute_remote_command_auth_exception(mock_paramiko_ssh_client, mock_paramiko_key):
    """Test handling of authentication errors."""
    hostname = "testhost"
    username = "wronguser"
    key_path = "/fake/key.pem"
    command = "ls"

    mock_client, _, _ = mock_paramiko_ssh_client
    # Configure the mock connect method to raise an AuthenticationException
    mock_client.connect.side_effect = paramiko.AuthenticationException("Auth failed")

    output, error, status = execute_remote_command(hostname, username, key_path, command)

    # Assertions
    assert status == -1 # Or your chosen error code for auth errors
    assert output is None
    assert "Error de autenticación" in error
    mock_client.exec_command.assert_not_called()
    mock_client.close.assert_called_once()

def test_execute_remote_command_timeout(mock_paramiko_ssh_client, mock_paramiko_key):
    """Test handling of command execution timeout."""
    hostname = "testhost"
    username = "testuser"
    key_path = "/fake/key.pem"
    command = "sleep 100"

    mock_client, _, _ = mock_paramiko_ssh_client
    # Configure exec_command to raise a TimeoutError (socket.timeout)
    mock_client.exec_command.side_effect = TimeoutError("Command timed out")

    output, error, status = execute_remote_command(hostname, username, key_path, command)

    # Assertions
    assert status == -2 # Or your chosen error code for command timeout
    assert output is None
    assert "Timeout ejecutando comando" in error
    mock_client.close.assert_called_once()

# Add more tests for edge cases:
# - Empty command
# - Different ports
# - Key with passphrase (if you implement that)
# - Handling of non-UTF8 output/error (using errors='replace')
