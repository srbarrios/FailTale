import pytest
import pytest_asyncio
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call

# --- Import the function to test ---
from mcp_server.ssh_executor import execute_remote_command_async
import asyncssh # Import the library itself for its exceptions
from asyncssh.misc import HostKeyNotVerifiable, PermissionDenied
from asyncssh.connection import ConnectionLost, DisconnectError

# --- Async Test Setup ---

@pytest_asyncio.fixture
def mock_ssh():
    """Fixture to mock asyncssh.connect and its returned objects."""
    mock_process = MagicMock(spec=asyncssh.SSHCompletedProcess)
    mock_process.stdout = "Async command output"
    mock_process.stderr = ""
    mock_process.exit_status = 0

    # Mock the connection object returned by connect
    mock_connection = AsyncMock(spec=asyncssh.SSHClientConnection) # Keep spec if possible
    mock_connection.run = AsyncMock(return_value=mock_process)
    mock_connection.close = MagicMock()
    mock_connection.wait_closed = AsyncMock()
    # --- FIX 1: Add is_closed attribute ---
    # Set it as a regular attribute returning False by default
    mock_connection.is_closed = False

    # Patch asyncssh.connect to be an awaitable mock returning our mock connection
    with patch('asyncssh.connect', new_callable=AsyncMock, return_value=mock_connection) as patched_connect:
        with patch('asyncio.sleep', new_callable=AsyncMock) as patched_sleep:
            yield patched_connect, mock_connection, mock_process, patched_sleep


# --- Async Test Cases ---

@pytest.mark.asyncio
async def test_execute_remote_command_async_success_first_try(mock_ssh):
    """Test successful async command execution on the first attempt (key auth)."""
    hostname = "async_testhost"
    username = "testuser"
    key_path = "/fake/async_key.pem"
    command = "echo 'hello async'"
    connect_timeout = 15
    cmd_timeout = 25

    mock_connect, mock_connection, mock_process, mock_sleep = mock_ssh

    output, error, status = await execute_remote_command_async(
        hostname, username, command, private_key_path=key_path,
        conn_timeout=connect_timeout, cmd_timeout=cmd_timeout, retries=1
    )

    mock_connect.assert_awaited_once_with(
        host=hostname, port=22, username=username, client_keys=[key_path],
        password=None, passphrase=None, known_hosts=None, connect_timeout=connect_timeout,
    )
    mock_connection.run.assert_awaited_once_with(command, check=False)
    assert status == 0
    assert output == "Async command output"
    assert error == ""
    mock_sleep.assert_not_awaited()
    mock_connection.close.assert_called_once()
    mock_connection.wait_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_remote_command_async_failure_status(mock_ssh):
    """Test command execution returning a non-zero exit status (no retry)."""
    hostname = "async_testhost"
    username = "testuser"
    key_path = "/fake/async_key.pem"
    command = "exit 1"

    mock_connect, mock_connection, mock_process, mock_sleep = mock_ssh

    mock_process.stdout = ""
    mock_process.stderr = "Async error message"
    mock_process.exit_status = 1

    output, error, status = await execute_remote_command_async(hostname, username, command, private_key_path=key_path)

    assert status == 1
    assert output == ""
    assert error == "Async error message"
    mock_connect.assert_awaited_once()
    mock_connection.run.assert_awaited_once_with(command, check=False)
    mock_sleep.assert_not_awaited()
    mock_connection.close.assert_called_once()
    mock_connection.wait_closed.assert_awaited_once()

@pytest.mark.asyncio
async def test_execute_remote_command_async_auth_error_no_retry(mock_ssh):
    """Test handling of async authentication errors (should not retry)."""
    hostname = "async_testhost"
    username = "wronguser"
    key_path = "/fake/async_key.pem"
    command = "ls"

    mock_connect, mock_connection, _, mock_sleep = mock_ssh

    error_message = "Permission denied"
    mock_connect.side_effect = PermissionDenied(error_message)

    output, error, status = await execute_remote_command_async(hostname, username, command, private_key_path=key_path, retries=2)

    assert status == -1
    assert output is None
    assert f"Authentication error: {error_message}" in error
    mock_connect.assert_awaited_once()
    mock_connection.run.assert_not_awaited()
    mock_sleep.assert_not_awaited()
    mock_connection.close.assert_not_called()
    mock_connection.wait_closed.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_remote_command_async_connection_error_retry_success(mock_ssh):
    """Test retry on ConnectionError, succeeding on the second attempt."""
    hostname = "flaky_host"
    username = "testuser"
    key_path = "/fake/async_key.pem"
    command = "uptime"
    retry_delay = 1.5

    mock_connect, mock_connection, mock_process, mock_sleep = mock_ssh

    mock_connect.side_effect = [
        ConnectionLost("Network is unreachable"), # First call fails
        mock_connection, # Second call returns the mocked connection
    ]

    mock_process.stdout = " 10:00 up 1 day, 2 users"
    mock_process.stderr = ""
    mock_process.exit_status = 0

    output, error, status = await execute_remote_command_async(
        hostname, username, command, private_key_path=key_path, retries=1, retry_delay=retry_delay
    )

    assert status == 0
    assert output == "10:00 up 1 day, 2 users"
    assert error == ""
    assert mock_connect.await_count == 2
    mock_sleep.assert_awaited_once_with(retry_delay)
    mock_connection.run.assert_awaited_once_with(command, check=False)
    mock_connection.close.assert_called_once()
    mock_connection.wait_closed.assert_awaited_once()

@pytest.mark.asyncio
async def test_execute_remote_command_async_timeout_error_retry_success(mock_ssh):
    """Test retry on asyncio.TimeoutError during run, succeeding on the second attempt."""
    hostname = "slow_command_host"
    username = "testuser"
    key_path = "/fake/async_key.pem"
    command = "long_running_script"
    cmd_timeout = 5
    retry_delay = 1.0

    mock_connect, mock_connection, mock_process, mock_sleep = mock_ssh

    # Need two separate connection mocks if connect succeeds both times but run fails first
    mock_connection_retry = AsyncMock(spec=asyncssh.SSHClientConnection)
    mock_connection_retry.run = AsyncMock(return_value=mock_process) # Success on second run
    mock_connection_retry.close = MagicMock()
    mock_connection_retry.wait_closed = AsyncMock()
    # --- FIX 1: Add is_closed attribute ---
    mock_connection_retry.is_closed = False

    mock_connection.run.side_effect = asyncio.TimeoutError # First run times out
    mock_connect.side_effect = [mock_connection, mock_connection_retry] # Return mocks sequentially

    mock_process.stdout = "Script finished"
    mock_process.stderr = ""
    mock_process.exit_status = 0

    output, error, status = await execute_remote_command_async(
        hostname, username, command, private_key_path=key_path,
        cmd_timeout=cmd_timeout, retries=1, retry_delay=retry_delay
    )

    assert status == 0
    assert output == "Script finished"
    assert error == ""
    assert mock_connect.await_count == 2
    mock_connection.run.assert_awaited_once_with(command, check=False)
    mock_connection_retry.run.assert_awaited_once_with(command, check=False)
    mock_sleep.assert_awaited_once_with(retry_delay)
    mock_connection.close.assert_called_once()
    mock_connection_retry.close.assert_called_once()
    mock_connection_retry.wait_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_remote_command_async_generic_error_retry_success(mock_ssh):
    """Test retry on generic Exception during connect, succeeding on the second attempt."""
    hostname = "weird_host"
    username = "testuser"
    key_path = "/fake/async_key.pem"
    command = "ls"
    retry_delay = 0.5

    mock_connect, mock_connection, mock_process, mock_sleep = mock_ssh

    generic_error = ValueError("Something unexpected happened")
    mock_connect.side_effect = [
        generic_error, # First call fails
        mock_connection, # Second call returns the mocked connection
    ]

    mock_process.stdout = "file1 file2"
    mock_process.stderr = ""
    mock_process.exit_status = 0

    output, error, status = await execute_remote_command_async(
        hostname, username, command, private_key_path=key_path, retries=1, retry_delay=retry_delay
    )

    assert status == 0
    assert output == "file1 file2"
    assert error == ""
    assert mock_connect.await_count == 2
    mock_sleep.assert_awaited_once_with(retry_delay)
    mock_connection.run.assert_awaited_once_with(command, check=False)
    mock_connection.close.assert_called_once()
    mock_connection.wait_closed.assert_awaited_once()

@pytest.mark.asyncio
async def test_execute_remote_command_async_connection_error_max_retries(mock_ssh):
    """Test failure after exhausting retries for ConnectionError."""
    hostname = "always_unreachable"
    username = "testuser"
    key_path = "/fake/async_key.pem"
    command = "ping"
    retries = 2
    retry_delay = 0.1

    mock_connect, mock_connection, _, mock_sleep = mock_ssh

    error_message = "Network down"
    mock_connect.side_effect = ConnectionLost(error_message)

    output, error, status = await execute_remote_command_async(
        hostname, username, command, private_key_path=key_path, retries=retries, retry_delay=retry_delay
    )

    # Assertions
    # --- FIX 2: Expect -5 for connection errors after retries ---
    assert status == -5 # Specific code for connection errors after retries
    assert output is None
    assert "Connection error after retries" in error # Check function's specific message
    assert error_message in error
    assert mock_connect.await_count == retries + 1
    assert mock_sleep.await_count == retries
    mock_connection.run.assert_not_awaited()
    mock_connection.close.assert_not_called()
    mock_connection.wait_closed.assert_not_awaited()

# --- Password/Passphrase/Known Hosts Tests ---

@pytest.mark.asyncio
async def test_execute_remote_command_async_password_auth(mock_ssh):
    """Test successful execution using password authentication."""
    hostname = "pwd_host"
    username = "pwd_user"
    password = "secretpassword"
    command = "pwd"

    mock_connect, mock_connection, mock_process, mock_sleep = mock_ssh
    mock_process.stdout = "/home/pwd_user"

    output, error, status = await execute_remote_command_async(
        hostname, username, command, password=password, retries=0 # No key path
    )

    mock_connect.assert_awaited_once_with(
        host=hostname, port=22, username=username, client_keys=None,
        password=password, passphrase=None, known_hosts=None, connect_timeout=10,
    )
    assert status == 0
    assert output == "/home/pwd_user"
    assert error == ""
    mock_sleep.assert_not_awaited()
    mock_connection.close.assert_called_once()
    mock_connection.wait_closed.assert_awaited_once()

@pytest.mark.asyncio
async def test_execute_remote_command_async_passphrase_auth(mock_ssh):
    """Test successful execution using a key with a passphrase."""
    hostname = "passphrase_host"
    username = "keyuser"
    key_path = "/fake/encrypted_key.pem"
    passphrase = "keypassword"
    command = "hostname"

    mock_connect, mock_connection, mock_process, mock_sleep = mock_ssh
    mock_process.stdout = "passphrase_host"

    output, error, status = await execute_remote_command_async(
        hostname, username, command, private_key_path=key_path, passphrase=passphrase, retries=0
    )

    mock_connect.assert_awaited_once_with(
        host=hostname, port=22, username=username, client_keys=[key_path],
        password=None, passphrase=passphrase, known_hosts=None, connect_timeout=10,
    )
    assert status == 0
    assert output == "passphrase_host"
    assert error == ""
    mock_sleep.assert_not_awaited()
    mock_connection.close.assert_called_once()
    mock_connection.wait_closed.assert_awaited_once()

@pytest.mark.asyncio
async def test_execute_remote_command_async_known_hosts_success(mock_ssh):
    """Test successful execution when known_hosts file is provided."""
    hostname = "known_host"
    username = "testuser"
    key_path = "/fake/async_key.pem"
    known_hosts_path = "/fake/known_hosts"
    command = "id"

    mock_connect, mock_connection, mock_process, mock_sleep = mock_ssh
    mock_process.stdout = "uid=1000(testuser) gid=1000(testuser)"

    output, error, status = await execute_remote_command_async(
        hostname, username, command, private_key_path=key_path, known_hosts=known_hosts_path, retries=0
    )

    mock_connect.assert_awaited_once_with(
        host=hostname, port=22, username=username, client_keys=[key_path],
        password=None, passphrase=None, known_hosts=known_hosts_path, connect_timeout=10,
    )
    assert status == 0
    assert output == "uid=1000(testuser) gid=1000(testuser)"
    assert error == ""
    mock_sleep.assert_not_awaited()
    mock_connection.close.assert_called_once()
    mock_connection.wait_closed.assert_awaited_once()

@pytest.mark.asyncio
async def test_execute_remote_command_async_known_hosts_mismatch(mock_ssh):
    """Test handling of known_hosts mismatch (simulated via HostKeyNotVerifiable)."""
    hostname = "unknown_host"
    username = "testuser"
    key_path = "/fake/async_key.pem"
    known_hosts_path = "/fake/known_hosts"
    command = "ls"

    mock_connect, mock_connection, _, mock_sleep = mock_ssh

    error_message = "Host key verification failed."
    mock_connect.side_effect = HostKeyNotVerifiable(error_message)

    output, error, status = await execute_remote_command_async(
        hostname, username, command, private_key_path=key_path, known_hosts=known_hosts_path, retries=0
    )

    assert status == -6 # Code for host key errors
    assert output is None
    assert f"Host key verification failed: {error_message}" in error

    mock_connect.assert_awaited_once()
    mock_connection.run.assert_not_awaited()
    mock_sleep.assert_not_awaited()
    mock_connection.close.assert_not_called()
    mock_connection.wait_closed.assert_not_awaited()
