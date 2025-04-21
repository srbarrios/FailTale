import asyncio
import logging
from typing import Optional, Tuple

import asyncssh


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
    Asynchronously connects via SSH and executes a command with retries.
    Returns: (stdout, stderr, exit_status)
    """
    attempt = 0
    while attempt <= retries:
        try:
            client_keys = [private_key_path] if private_key_path else None

            async with asyncssh.connect(
                host=hostname,
                port=port,
                username=username,
                client_keys=client_keys,
                password=password,
                passphrase=passphrase,
                known_hosts=known_hosts or None,
                connect_timeout=conn_timeout,
            ) as conn:
                logging.info(f"[{hostname}] Connected, running: {command}")
                result = await asyncio.wait_for(conn.run(command, check=False), timeout=cmd_timeout)

                return result.stdout.strip(), result.stderr.strip(), result.exit_status

        except asyncssh.AuthError:
            logging.error(f"[{hostname}] AUTH_ERROR. Check credentials.")
            return None, "Authentication error", -1
        except asyncssh.ConnectionError as e:
            logging.error(f"[{hostname}] CONNECTION_ERROR: {e}")
            if attempt == retries:
                return None, f"Connection error: {e}", -1
        except asyncio.TimeoutError:
            logging.error(f"[{hostname}] TIMEOUT (limit: {cmd_timeout}s)")
            if attempt == retries:
                return None, f"Command timeout (limit: {cmd_timeout}s)", -2
        except Exception as e:
            logging.exception(f"[{hostname}] UNEXPECTED_ERROR: {e}")
            if attempt == retries:
                return None, f"Unexpected error: {e}", -3

        attempt += 1
        logging.warning(f"[{hostname}] Retrying in {retry_delay}s (attempt {attempt}/{retries})...")
        await asyncio.sleep(retry_delay)

    return None, "Max retries reached", -4
