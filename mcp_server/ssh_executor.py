import asyncio
import logging
from typing import Optional, Tuple

import asyncssh # Import the library itself for its exceptions
# Import specific exceptions for clarity and potentially different handling
from asyncssh.misc import HostKeyNotVerifiable, PermissionDenied
from asyncssh.connection import ConnectionLost, DisconnectError
# ConnectionRefusedError is a built-in exception, usually caught by OSError


async def execute_remote_command_async(
        hostname: str,
        username: str,
        command: str,
        private_key_path: Optional[str] = None,
        passphrase: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 22,
        conn_timeout: int = 10,
        cmd_timeout: int = 30,
        known_hosts: Optional[str] = None, # Path to known_hosts file or None to disable check
        retries: int = 2,
        retry_delay: float = 3.0,
) -> Tuple[Optional[str], Optional[str], int]:
    """
    Asynchronously connects via SSH using asyncssh and executes a command with retries.

    Handles common connection, authentication, timeout, and host key errors.

    Args:
        hostname: Target hostname or IP address.
        username: SSH username.
        command: The command string to execute on the remote host.
        private_key_path: Optional path to the SSH private key file.
        passphrase: Optional passphrase for the private key.
        password: Optional password for password authentication (use instead of key).
        port: SSH port number.
        conn_timeout: Connection timeout in seconds.
        cmd_timeout: Command execution timeout in seconds.
        known_hosts: Path to a known_hosts file for host key verification.
                     Set to None to disable host key checking (INSECURE, use with caution).
        retries: Number of retries upon encountering recoverable errors (timeouts, connection issues).
        retry_delay: Delay in seconds between retries.

    Returns:
        A tuple containing:
        - Optional[str]: stdout from the command, stripped of whitespace, or None on error.
        - Optional[str]: stderr from the command, stripped of whitespace, or None on error.
        - int: Exit status code of the command. Negative values indicate specific execution errors:
            - -1: Authentication error (PermissionDenied)
            - -2: Command execution timeout (asyncio.TimeoutError)
            - -3: Unexpected error during connection or execution (generic Exception, or asyncssh.Error)
            - -4: Max retries reached after recoverable errors
            - -5: Connection error (ConnectionLost, DisconnectError, OSError including ConnectionRefusedError)
            - -6: Host key verification failed (HostKeyNotVerifiable)
    """
    attempt = 0
    last_error: Optional[Exception] = None # Store the last exception for logging/return message

    while attempt <= retries:
        conn = None # Define conn outside try block for potential cleanup if needed (though async with handles it)
        try:
            # Prepare client keys list (asyncssh expects a list or None)
            client_keys = [private_key_path] if private_key_path else None

            # Establish SSH connection using async context manager
            conn = await asyncssh.connect( # Await the connect coroutine directly
                host=hostname,
                port=port,
                username=username,
                client_keys=client_keys,
                password=password,
                passphrase=passphrase,
                known_hosts=known_hosts, # Pass path directly, None disables checking
                connect_timeout=conn_timeout,
            )
            # Connection successful, enter context manually if needed or just use conn
            logging.info(f"[{hostname}] SSH connection successful (attempt {attempt+1}). Running command: {command}")

            # Execute the command, wait for completion with timeout
            # check=False ensures we get stderr/status even if command fails
            result = await asyncio.wait_for(
                conn.run(command, check=False),
                timeout=cmd_timeout
            )

            logging.info(f"[{hostname}] Command finished with exit status: {result.exit_status}")
            # Close connection before returning successful result
            conn.close()
            await conn.wait_closed() # Ensure connection is fully closed
            return result.stdout.strip(), result.stderr.strip(), result.exit_status

        # --- Specific Exception Handling ---
        except PermissionDenied as e:
            logging.error(f"[{hostname}] AUTH_ERROR (Permission Denied): {e}. Check credentials. No retry.")
            # Ensure connection is closed if partially opened (though unlikely here)
            if conn and not conn.is_closed: conn.close()
            return None, f"Authentication error: {e}", -1

        except HostKeyNotVerifiable as e:
            logging.error(f"[{hostname}] HOST_KEY_ERROR: {e}. Verify host key in known_hosts file: {known_hosts}. No retry.")
            if conn and not conn.is_closed: conn.close()
            return None, f"Host key verification failed: {e}", -6

        # --- Corrected: Removed asyncssh.ConnectionRefused ---
        except (ConnectionLost, DisconnectError, OSError) as e:
            logging.warning(f"[{hostname}] CONNECTION_ERROR (Attempt {attempt+1}/{retries+1}): {e}")
            last_error = e
            if attempt == retries:
                logging.error(f"[{hostname}] Max retries reached for connection errors.")
                if conn and not conn.is_closed: conn.close()
                return None, f"Connection error after retries: {e}", -5

        except asyncio.TimeoutError:
            logging.warning(f"[{hostname}] COMMAND_TIMEOUT (Attempt {attempt+1}/{retries+1}, limit: {cmd_timeout}s)")
            last_error = asyncio.TimeoutError(f"Command timed out after {cmd_timeout}s")
            if attempt == retries:
                logging.error(f"[{hostname}] Max retries reached for command timeout.")
                if conn and not conn.is_closed: conn.close()
                return None, f"Command timeout after retries (limit: {cmd_timeout}s)", -2

        except asyncssh.Error as e: # Catch other asyncssh specific errors
            logging.warning(f"[{hostname}] ASYNCSSH_ERROR (Attempt {attempt+1}/{retries+1}): {e}")
            last_error = e
            if attempt == retries:
                logging.error(f"[{hostname}] Max retries reached for asyncssh error: {e}")
                if conn and not conn.is_closed: conn.close()
                return None, f"SSH error after retries: {e}", -3 # Return -3 for asyncssh.Error

        except Exception as e: # Catch any other unexpected errors
            logging.exception(f"[{hostname}] UNEXPECTED_ERROR (Attempt {attempt+1}/{retries+1}): {e}")
            last_error = e
            if attempt == retries:
                logging.error(f"[{hostname}] Max retries reached for unexpected error: {e}")
                if conn and not conn.is_closed: conn.close()
                return None, f"Unexpected error after retries: {e}", -3 # Return -3 for generic Exception

        # If we are here, an error occurred but we can retry
        attempt += 1
        if attempt <= retries:
            logging.info(f"[{hostname}] Retrying in {retry_delay}s...")
            # Ensure connection is closed before sleeping/retrying
            if conn and not conn.is_closed:
                conn.close()
                # Don't necessarily need to wait_closed() before retry sleep
            await asyncio.sleep(retry_delay)

    # This point should only be reached if all retries failed for recoverable errors
    logging.error(f"[{hostname}] Failed after {retries} retries. Last error: {last_error}")
    return None, f"Max retries reached. Last error: {last_error}", -4
