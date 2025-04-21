import logging
from typing import List, Dict, Any, Optional

import requests


class McpApiClient:
    """
    A simple client to interact with the local MCP Server API.
    """
    def __init__(self, base_url: str, timeout: int = 60):
        """
        Initializes the client.

        Args:
            base_url (str): The base URL of the MCP Server (e.g., "http://localhost:5050").
            timeout (int): Request timeout in seconds.
        """
        if not base_url:
            raise ValueError("Base URL for MCP Server is required.")
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.collect_endpoint = f"{self.base_url}/mcp/v1/collect"
        self.analyze_endpoint = f"{self.base_url}/mcp/v1/analyze"
        logging.debug(f"MCP API Client initialized for URL: {self.base_url}")

    def _make_request(self, method: str, url: str, json_payload: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Helper method to make HTTP requests and handle common errors."""
        try:
            logging.debug(f"Sending {method} request to {url} with payload: {json_payload}")
            response = requests.request(method, url, json=json_payload, timeout=self.timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logging.debug(f"Received successful response ({response.status_code}) from {url}")
            return response.json()
        except requests.exceptions.Timeout:
            logging.error(f"Timeout error connecting to MCP server at {url} (limit: {self.timeout}s)")
            # Re-raise or return None/Error indicator based on desired handling
            raise ConnectionError(f"Timeout connecting to MCP server at {url}")
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error to MCP server at {url}: {e}")
            raise ConnectionError(f"Could not connect to MCP server at {url}: {e}")
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error from MCP server at {url}: {e.response.status_code} {e.response.reason}")
            try:
                # Try to get error details from response body
                error_details = e.response.json()
                logging.error(f"Server error details: {error_details}")
            except ValueError: # Handle cases where response body is not JSON
                 logging.error(f"Server response body: {e.response.text}")
            raise ConnectionError(f"HTTP {e.response.status_code} error from MCP server at {url}")
        except Exception as e:
            logging.exception(f"An unexpected error occurred during request to {url}: {e}")
            raise # Re-raise unexpected errors

    def collect_data(self, hosts: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Sends a request to the /collect endpoint.

        Args:
            hosts (List[Dict[str, str]]): List of host dictionaries,
                                          e.g., [{"hostname": "h1", "role": "r1"}, ...].

        Returns:
            Optional[Dict[str, Any]]: The JSON response from the server (collected data)
                                      or None/raises Exception on error.
        """
        if not hosts:
            logging.warning("Collect data called with empty host list.")
            return None

        payload = {"hosts": hosts}
        return self._make_request("POST", self.collect_endpoint, json_payload=payload)

    def analyze_data(self, collected_data: Dict[str, Any], failure_summary: str) -> Optional[Dict[str, Any]]:
        """
        Sends a request to the /analyze endpoint.

        Args:
            collected_data (Dict[str, Any]): The data previously collected by the server.
            failure_summary (str): A summary of the test failure reason.

        Returns:
            Optional[Dict[str, Any]]: The JSON response from the server (AI hint)
                                      or None/raises Exception on error.
        """
        if not collected_data:
            logging.warning("Analyze data called with empty collected_data.")
            return None

        payload = {
            "collected_data": collected_data,
            "failure_summary": failure_summary
        }
        return self._make_request("POST", self.analyze_endpoint, json_payload=payload)

# --- Example Usage (can be run standalone for basic testing) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Ensure your MCP server is running locally on port 5050 before running this
    mcp_server_test_url = "http://localhost:5050"
    client = McpApiClient(base_url=mcp_server_test_url)

    test_hosts = [
        {"hostname": "localhost", "role": "server"}, # Replace with actual testable hosts if needed
        # {"hostname": "anotherhost", "role": "minion"}
    ]

    print(f"--- Testing Collection from {mcp_server_test_url} ---")
    try:
        data = client.collect_data(hosts=test_hosts)
        if data:
            print("Collection successful:")
            import json
            print(json.dumps(data, indent=2))

            print(f"\n--- Testing Analysis from {mcp_server_test_url} ---")
            try:
                analysis = client.analyze_data(
                    collected_data=data,
                    failure_summary="Example failure: Assertion failed in test_logic"
                )
                if analysis:
                    print("Analysis successful:")
                    print(json.dumps(analysis, indent=2))
                else:
                    print("Analysis returned no data.")
            except Exception as e:
                print(f"Analysis failed: {e}")

        else:
            print("Collection returned no data.")
    except Exception as e:
        print(f"Collection failed: {e}")
