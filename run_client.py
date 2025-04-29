# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import requests
import logging
import yaml
import os
import argparse
from typing import List, Dict, Any, Optional

class FailTaleClient:
    """
    A client to interact with the Server API.
    Loads target hosts from a specified YAML environment file.
    """
    def __init__(self, base_url: str, environment_file: Optional[str] = None, timeout: int = 180):
        """
        Initializes the client. Optionally loads hosts if environment_file is provided.

        Args:
            base_url (str): The base URL of the Server (e.g., "http://localhost:5050").
            environment_file (Optional[str]): Path to the YAML file containing the host list.
                                              If None, hosts are not loaded automatically.
            timeout (int): Request timeout in seconds.
        """
        if not base_url:
            raise ValueError("Base URL for Server is required.")
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.collect_endpoint = f"{self.base_url}/v1/collect"
        self.analyze_endpoint = f"{self.base_url}/v1/analyze"
        self.environment_file = environment_file
        self.hosts = None # Initialize hosts to None

        if self.environment_file:
            self.hosts = self._load_hosts_from_yaml() # Load hosts if file is provided
            if self.hosts:
                logging.debug(f"Loaded {len(self.hosts)} hosts from {self.environment_file}")
            else:
                logging.warning(f"Could not load hosts from {self.environment_file}. Collection might not work unless hosts are passed explicitly.")
        else:
            logging.warning("No environment file provided at initialization. Hosts not loaded.")


        logging.debug(f"Client initialized for URL: {self.base_url}")


    def _load_hosts_from_yaml(self) -> Optional[List[Dict[str, str]]]:
        """Loads the host list from the specified YAML file."""
        if not self.environment_file or not os.path.exists(self.environment_file):
            logging.error(f"Environment file not found or not specified: {self.environment_file}")
            return None
        try:
            with open(self.environment_file, 'r') as f:
                data = yaml.safe_load(f)
                # Expecting a top-level 'hosts' key with a list of dicts
                if isinstance(data, dict) and 'hosts' in data and isinstance(data['hosts'], list):
                    valid_hosts = []
                    for host in data['hosts']:
                        if isinstance(host, dict) and 'hostname' in host and 'role' in host:
                            valid_hosts.append({
                                'hostname': str(host['hostname']),
                                'role': str(host['role']),
                                'mandatory': host.get('mandatory', False)
                            })
                        else:
                            logging.warning(f"Skipping invalid host entry in {self.environment_file}: {host}")
                    return valid_hosts
                else:
                    logging.error(f"Invalid format in {self.environment_file}. Expected a 'hosts' key with a list.")
                    return None
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {self.environment_file}: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error loading hosts from {self.environment_file}: {e}")
            return None

    def _make_request(self, method: str, url: str, json_payload: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Helper method to make HTTP requests and handle common errors."""
        try:
            logging.debug(f"Sending {method} request to {url} with payload: {json_payload}")
            response = requests.request(method, url, json=json_payload, timeout=self.timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logging.debug(f"Received successful response ({response.status_code}) from {url}")
            return response.json()
        except requests.exceptions.Timeout:
            logging.error(f"Timeout error connecting to server at {url} (limit: {self.timeout}s)")
            raise ConnectionError(f"Timeout connecting to server at {url}")
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error to server at {url}: {e}")
            raise ConnectionError(f"Could not connect to server at {url}: {e}")
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error from server at {url}: {e.response.status_code} {e.response.reason}")
            try:
                error_details = e.response.json()
                logging.error(f"Server error details: {error_details}")
            except ValueError:
                logging.error(f"Server response body: {e.response.text}")
            raise ConnectionError(f"HTTP {e.response.status_code} error from server at {url}")
        except Exception as e:
            logging.exception(f"An unexpected error occurred during request to {url}: {e}")
            raise

    def collect_data(self, hosts: Optional[List[Dict[str, str]]] = None, test_report: str = None) -> Optional[Dict[str, Any]]:
        """
        Sends a request to the /collect endpoint. Uses hosts loaded during
        initialization if the `hosts` argument is not provided.

        Args:
            hosts (Optional[List[Dict[str, str]]]): Explicit list of host dictionaries
                                                   to collect data from. Overrides hosts
                                                   loaded from the environment file.
                                                   e.g., [{"hostname": "h1", "role": "r1"}, ...].
            test_report (str): Test report for the collection request.

        Returns:
            Optional[Dict[str, Any]]: The JSON response from the server (collected data)
                                      or None/raises Exception on error or if no hosts are available.
        """
        target_hosts = hosts if hosts is not None else self.hosts

        if not target_hosts:
            logging.error("Cannot collect data: No hosts provided or loaded.")
            return None # Or raise an exception

        payload = {"hosts": target_hosts, "test_report": test_report}
        return self._make_request("POST", self.collect_endpoint, json_payload=payload)

    def analyze_data(self, collected_data: Dict[str, Any], test_report: str = None, test_failure: str = None) -> Optional[Dict[str, Any]]:
        """
        Sends a request to the /analyze endpoint.

        Args:
            collected_data (Dict[str, Any]): The data previously collected by the server.
            test_report (str): The test report.
            test_failure (str): A summary of the test failure reason.

        Returns:
            Optional[Dict[str, Any]]: The JSON response from the server (AI hint)
                                      or None/raises Exception on error.
        """
        if not collected_data:
            logging.warning("Analyze data called with empty collected_data.")
            return None

        payload = {
            "collected_data": collected_data,
            "test_report": test_report,
            "test_failure": test_failure
        }
        return self._make_request("POST", self.analyze_endpoint, json_payload=payload)

# --- Example Usage (Runnable from command line) ---
if __name__ == "__main__":
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Collects debug data from hosts defined in a YAML file, and analyze it through an AI agent"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to the environment YAML file containing the list of hosts."
    )
    parser.add_argument(
        "-s", "--server-url",
        default="http://localhost:5050",
        help="Base URL of the server."
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    parser.add_argument(
        "-r", "--test-report",
        required=True,
        help="Test report file path."
    )
    parser.add_argument(
        "-f", "--test-failure",
        required=True,
        help="Test failure file path."
    )
    args = parser.parse_args()
    # --- End Argument Parser Setup ---

    # Configure logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Read test report from file
    test_report = None
    if args.test_report:
        try:
            with open(args.test_report, 'r') as f:
                test_report = f.read()
        except Exception as e:
            logging.error(f"Error reading test report file: {e}")
            exit(1)

    # Read test failure from file
    test_failure = None
    if args.test_failure:
        try:
            with open(args.test_failure, 'r') as f:
                test_failure = f.read()
        except Exception as e:
            logging.error(f"Error reading test test_failure file: {e}")
            exit(1)

    # Initialize the client, passing the config file path from arguments
    try:
        client = FailTaleClient(
            base_url=args.server_url,
            environment_file=args.config,
            timeout=args.timeout
        )
    except ValueError as e:
        logging.error(f"Error initializing client: {e}")
        exit(1)

    if not client.hosts:
        logging.error(f"Failed to load hosts from {args.config}. Cannot proceed.")
        exit(1)

    try:
        # Call collect_data - it will use the hosts loaded from the file
        data = client.collect_data(test_report=test_report)
        if data:
            try:
                # Use a generic failure summary for standalone testing
                analysis = client.analyze_data(
                    collected_data=data,
                    test_report=test_report,
                    test_failure=test_failure
                )
                if analysis:
                    print(analysis["root_cause_hint"])
                else:
                    logging.warning("Analysis returned no data or failed.")
            except Exception as e:
                logging.error(f"Analysis failed: {e}")

        else:
            logging.warning("Collection returned no data or failed.")
    except ConnectionError as e:
        logging.error(f"Connection Error during API call: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during API calls: {e}")
