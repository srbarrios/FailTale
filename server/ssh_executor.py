# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import asyncio
import logging
from typing import Optional, Tuple

import asyncssh
from asyncssh.connection import ConnectionLost, DisconnectError
from asyncssh.misc import HostKeyNotVerifiable, PermissionDenied


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
        known_hosts: Optional[str] = None,
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
        password: Optional password for password authentication (alternative to key).
        port: SSH port number.
        conn_timeout: Connection timeout in seconds.
        cmd_timeout: Command execution timeout in seconds.
        known_hosts: Path to a known_hosts file for host key verification (None disables checking — INSECURE).
        retries: Number of retries upon encountering recoverable errors.
        retry_delay: Delay in seconds between retries.

    Returns:
        A tuple (stdout, stderr, exit_status):
            - stdout: Command output (stripped) or None on error.
            - stderr: Command error output (stripped) or error message.
            - exit_status:
                - 0 or other positive exit code for command result,
                - negative values for specific internal errors:
                    - -1: Authentication error (PermissionDenied)
                    - -2: Command execution timeout
                    - -3: Unexpected error during connection or execution
                    - -4: Max retries reached
                    - -5: Connection error (ConnectionLost, DisconnectError, OSError)
                    - -6: Host key verification failed
    """
    attempt = 0
    last_error: Optional[Exception] = None

    while attempt <= retries:
        conn = None
        try:
            client_keys = [private_key_path] if private_key_path else None

            conn = await asyncssh.connect(
                host=hostname,
                port=port,
                username=username,
                client_keys=client_keys,
                password=password,
                passphrase=passphrase,
                known_hosts=known_hosts,
                connect_timeout=conn_timeout,
            )

            logging.debug(f"[{hostname}] SSH connection established (attempt {attempt+1}). Executing command: {command}")

            result = await asyncio.wait_for(
                conn.run(command, check=False),
                timeout=cmd_timeout,
            )

            logging.debug(f"[{hostname}] Command completed with exit status {result.exit_status}")

            conn.close()
            await conn.wait_closed()
            return result.stdout.strip(), result.stderr.strip(), result.exit_status

        except PermissionDenied as e:
            logging.error(f"[{hostname}] Authentication failed: {e}. No retry.")
            if conn and not conn.is_closed:
                conn.close()
            return None, f"Authentication error: {e}", -1

        except HostKeyNotVerifiable as e:
            logging.error(f"[{hostname}] Host key verification failed: {e}. No retry.")
            if conn and not conn.is_closed:
                conn.close()
            return None, f"Host key verification failed: {e}", -6

        except (ConnectionLost, DisconnectError, OSError) as e:
            logging.warning(f"[{hostname}] Connection error (attempt {attempt+1}/{retries+1}): {e}")
            last_error = e
            if attempt == retries:
                logging.error(f"[{hostname}] Max retries reached due to connection errors.")
                if conn and not conn.is_closed:
                    conn.close()
                return None, f"Connection error after retries: {e}", -5

        except asyncio.TimeoutError:
            logging.warning(f"[{hostname}] Command timeout (attempt {attempt+1}/{retries+1}, limit: {cmd_timeout}s)")
            last_error = asyncio.TimeoutError(f"Command timed out after {cmd_timeout}s")
            if attempt == retries:
                logging.error(f"[{hostname}] Max retries reached due to command timeouts.")
                if conn and not conn.is_closed:
                    conn.close()
                return None, f"Command timeout after retries (limit: {cmd_timeout}s)", -2

        except asyncssh.Error as e:
            logging.warning(f"[{hostname}] AsyncSSH error (attempt {attempt+1}/{retries+1}): {e}")
            last_error = e
            if attempt == retries:
                logging.error(f"[{hostname}] Max retries reached due to AsyncSSH errors.")
                if conn and not conn.is_closed:
                    conn.close()
                return None, f"SSH error after retries: {e}", -3

        except Exception as e:
            logging.exception(f"[{hostname}] Unexpected error (attempt {attempt+1}/{retries+1}): {e}")
            last_error = e
            if attempt == retries:
                logging.error(f"[{hostname}] Max retries reached due to unexpected errors.")
                if conn and not conn.is_closed:
                    conn.close()
                return None, f"Unexpected error after retries: {e}", -3

        finally:
            if conn and not conn.is_closed:
                conn.close()

        attempt += 1
        if attempt <= retries:
            logging.warning(f"[{hostname}] Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)

    logging.error(f"[{hostname}] All retries exhausted. Last error: {last_error}")
    return None, f"Max retries reached. Last error: {last_error}", -4
