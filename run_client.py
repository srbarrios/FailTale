# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import requests
import logging
import yaml
import os
import argparse
import json
import base64
from typing import List, Dict, Any, Optional

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout, force=True)
log = logging.getLogger(__name__)


class FailTaleClient:
    """
    A client to interact with the Server API.
    Loads target hosts from a specified YAML environment file.
    """
    def __init__(self, base_url: str, environment_file: Optional[str] = None, timeout: int = 600):
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
        self.hosts = None

        if self.environment_file:
            self.hosts = self._load_hosts_from_yaml()
            if self.hosts:
                log.debug(f"Loaded {len(self.hosts)} hosts from {self.environment_file}")
            else:
                log.warning(f"Could not load hosts from {self.environment_file}. Collection might not work unless hosts are passed explicitly.")
        else:
            log.warning("No environment file provided at initialization. Hosts not loaded.")

        log.debug(f"FailTaleClient initialized for URL: {self.base_url}")


    def _load_hosts_from_yaml(self) -> Optional[List[Dict[str, str]]]:
        """Loads the host list from the specified YAML file."""
        if not self.environment_file or not os.path.exists(self.environment_file):
            log.error(f"Environment file not found or not specified: {self.environment_file}")
            return None
        try:
            with open(self.environment_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
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
                    log.error(f"Invalid format in {self.environment_file}. Expected a 'hosts' key with a list.")
                    return None
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML file {self.environment_file}: {e}")
            return None
        except Exception as e:
            log.exception(f"Unexpected error loading hosts from {self.environment_file}: {e}")
            return None

    def _make_request(self, method: str, url: str, json_payload: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Helper method to make HTTP requests and handle common errors."""
        try:
            log.debug(f"Sending {method} request to {url} with payload: {json.dumps(json_payload, indent=2)}")
            response = requests.request(method, url, json=json_payload, timeout=self.timeout)
            log.debug(f"Received raw response status {response.status_code} from {url}. Body: {response.text[:500]}...")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            log.error(f"Timeout error connecting to server at {url} (limit: {self.timeout}s)")
            raise ConnectionError(f"Timeout connecting to server at {url}")
        except requests.exceptions.ConnectionError as e:
            log.error(f"Connection error to server at {url}: {e}")
            raise ConnectionError(f"Could not connect to server at {url}: {e}")
        except requests.exceptions.HTTPError as e:
            log.error(f"HTTP error from server at {url}: {e.response.status_code} {e.response.reason}")
            try:
                error_details = e.response.json()
                log.error(f"Server error details: {error_details}")
            except ValueError:
                log.error(f"Server response body: {e.response.text}")
            raise ConnectionError(f"HTTP {e.response.status_code} error from server at {url}")
        except json.JSONDecodeError as e:
            log.error(f"Failed to decode JSON response from {url}: {e}.")
            raise ValueError(f"Invalid JSON response from server at {url}")
        except Exception as e:
            log.exception(f"An unexpected error occurred during request to {url}: {e}")
            raise

    def collect_data(self, hosts: Optional[List[Dict[str, str]]] = None, test_report: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Sends a request to the /v1/collect endpoint. Uses hosts loaded during
        initialization if the `hosts` argument is not provided.
        """
        target_hosts = hosts if hosts is not None else self.hosts

        if not target_hosts:
            log.error("Cannot collect data: No hosts provided or loaded.")
            return {"error": "No hosts available for collection"}
        if not test_report:
            log.error("Cannot collect data: test_report is missing.")
            return {"error": "test_report is required for collection"}

        payload = {"hosts": target_hosts, "test_report": test_report}
        return self._make_request("POST", self.collect_endpoint, json_payload=payload)

    def analyze_data(self,
                     collected_data: Dict[str, Any],
                     page_html: Optional[str] = None,
                     test_report: Optional[str] = None,
                     test_failure: Optional[str] = None,
                     screenshot_b64: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Sends a request to the /v1/analyze endpoint.
        """
        if not collected_data:
            log.warning("Analyze data called with empty collected_data.")
            return {"error": "collected_data cannot be empty for analysis"}

        payload = {
            "collected_data": collected_data,
            "page_html": page_html or "",
            "test_report": test_report or "N/A",
            "test_failure": test_failure or "N/A",
            "screenshot": screenshot_b64 or ""
        }
        return self._make_request("POST", self.analyze_endpoint, json_payload=payload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collects debug data from hosts defined in a YAML file, and analyze it through the FailTale server."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to the environment YAML file (client-side) containing the list of hosts."
    )
    parser.add_argument(
        "-s", "--server-url",
        default="http://localhost:5050",
        help="Base URL of the FailTale server."
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds for API calls."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging."
    )
    parser.add_argument(
        "-r", "--test-report",
        required=True,
        help="Path to the Gherkin test report file."
    )
    parser.add_argument(
        "-f", "--test-failure",
        required=True,
        help="Path to the file containing the test failure message."
    )
    parser.add_argument(
        "--screenshot",
        required=False,
        help="Path to the screenshot PNG/JPEG file taken at the time of failure."
    )
    args = parser.parse_args()

    def read_file_content(file_path, binary_mode=False):
        content = None
        if file_path:
            try:
                mode = 'rb' if binary_mode else 'r'
                encoding = None if binary_mode else 'utf-8'
                with open(file_path, mode, encoding=encoding) as f:
                    content = f.read()
                if not content:
                    log.warning(f"'{file_path}' is empty.")
            except Exception as e:
                log.error(f"Error reading '{file_path}': {e}")
        return content

    test_report_content = read_file_content(args.test_report)
    test_failure_content = read_file_content(args.test_failure)

    screenshot_b64_content = None
    if args.screenshot:
        screenshot_bytes = read_file_content(args.screenshot, binary_mode=True)
        if screenshot_bytes:
            screenshot_b64_content = base64.b64encode(screenshot_bytes).decode('utf-8')
            log.info(f"Loaded and base64 encoded screenshot from {args.screenshot}")

    if not test_report_content:
        log.critical(f"Test report content from '{args.test_report}' is missing or empty. Exiting.")
        exit(1)
    if not test_failure_content:
        log.critical(f"Test failure content from '{args.test_failure}' is missing or empty. Exiting.")
        exit(1)

    try:
        client = FailTaleClient(
            base_url=args.server_url,
            environment_file=args.config,
            timeout=args.timeout
        )
    except ValueError as e:
        log.error(f"Error initializing client: {e}")
        exit(1)

    if not client.hosts:
        log.error(f"Failed to load hosts from {args.config} or no hosts defined. Cannot proceed.")
        exit(1)

    collected_data_from_server = None
    try:
        log.info(f"Requesting data collection from server: {args.server_url}...")
        collected_data_from_server = client.collect_data(test_report=test_report_content)

        if collected_data_from_server and not collected_data_from_server.get("error"):
            log.info("Data collection successful.")
            log.debug(f"Collected data: {json.dumps(collected_data_from_server, indent=2)}")
        elif collected_data_from_server and collected_data_from_server.get("error"):
            log.error(f"Data collection failed with server error: {collected_data_from_server.get('error')}")
        else:
            log.warning("Collection returned no data or an unexpected response.")

    except ConnectionError as e:
        log.error(f"Connection error during data collection: {e}")
        exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred during data collection: {e}")
        exit(1)

    if collected_data_from_server and not collected_data_from_server.get("error"):
        try:
            log.info(f"Requesting data analysis from server: {args.server_url}...")
            analysis_result = client.analyze_data(
                collected_data=collected_data_from_server,
                test_report=test_report_content,
                test_failure=test_failure_content,
                screenshot_b64=screenshot_b64_content
            )
            if analysis_result and "root_cause_hint" in analysis_result:
                print(analysis_result["root_cause_hint"])
            elif analysis_result and "error" in analysis_result:
                log.error(f"Analysis failed with server error: {analysis_result['error']}")
            else:
                log.warning("Analysis returned no data or an unexpected response.")
        except ConnectionError as e:
            log.error(f"Connection error during analysis: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred during analysis: {e}")
    else:
        log.info("Skipping analysis step due to issues in data collection or empty/error in collected data.")
