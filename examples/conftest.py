import logging
import os

import pytest

# Import the client functions/class from your example client module
from .mcp_client import McpApiClient  # Assuming you create a client class

# --- Configuration ---

# Read MCP Server URL from environment variable or use a default
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:5050")
# Flag to enable/disable data collection (useful for local runs vs CI)
ENABLE_MCP_COLLECTION = os.environ.get("ENABLE_MCP_COLLECTION", "true").lower() == "true"

# Global or fixture to store collected data temporarily if needed for reporting
_collected_debug_info = {}

# --- Pytest Hook Implementation ---

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Pytest hook executed after each test phase (setup, call, teardown).
    We are interested in failures during the 'call' phase.
    """
    # Let the default hook run first
    outcome = yield
    report = outcome.get_result()

    # Check if:
    # 1. MCP collection is enabled
    # 2. The failure happened during the main test execution ('call')
    # 3. The test actually failed
    if ENABLE_MCP_COLLECTION and report.when == 'call' and report.failed:
        logging.warning(f"\nTest Failed: {report.nodeid}. Triggering MCP data collection.")

        # --- Determine Relevant Hosts ---
        # This is highly dependent on your test setup. Examples:
        # Option 1: Use pytest markers (`@pytest.mark.hosts(...)`)
        marker_hosts = item.get_closest_marker("hosts")
        involved_hosts_list = []
        if marker_hosts:
            # Assuming marker format: @pytest.mark.hosts([{"hostname": "h1", "role": "r1"}, ...])
            involved_hosts_list = marker_hosts.args[0] if marker_hosts.args else []
        else:
            # Option 2: Environment variables, test naming conventions, etc.
            # Fallback to a default set if no specific hosts found?
            logging.warning(f"No @pytest.mark.hosts found for {report.nodeid}. Using default or skipping.")
            # involved_hosts_list = [{"hostname": "default-server", "role": "server"}] # Example fallback

        if not involved_hosts_list:
            logging.error("No relevant hosts identified. Skipping MCP collection.")
            return # Exit the hook early

        # --- Call the MCP Client ---
        try:
            client = McpApiClient(base_url=MCP_SERVER_URL)
            logging.info(f"Sending collection request to MCP Server at {MCP_SERVER_URL} for hosts: {involved_hosts_list}")

            # 1. Collect Data
            collected_data = client.collect_data(hosts=involved_hosts_list)

            if collected_data:
                logging.info("Data collected successfully.")
                # Store data for potential reporting later
                _collected_debug_info[report.nodeid] = {"collected": collected_data}

                # 2. (Optional) Analyze Data
                # Extract failure reason if possible (might need more robust parsing)
                failure_summary = report.longreprtext or "Unknown failure reason"
                try:
                    logging.info("Sending analysis request to MCP Server...")
                    analysis_result = client.analyze_data(
                        collected_data=collected_data,
                        failure_summary=failure_summary
                    )
                    if analysis_result and "ai_hint" in analysis_result:
                        logging.info("AI analysis received.")
                        _collected_debug_info[report.nodeid]["analysis"] = analysis_result
                    else:
                         logging.warning("Received empty or invalid analysis result.")

                except Exception as analysis_exc:
                    logging.error(f"Error during MCP analysis request: {analysis_exc}", exc_info=True)

            else:
                logging.warning("MCP Server returned no collected data.")

        except Exception as client_exc:
            logging.error(f"Error during MCP collection request: {client_exc}", exc_info=True)

# --- Reporting Integration (Example for pytest-html) ---
# This part requires the pytest-html plugin to be installed.
# You might need to adapt it based on the plugin version and your needs.

@pytest.hookimpl(optionalhook=True)
def pytest_html_results_table_header(cells):
    """Add a header cell for the debug info."""
    try:
        from py.xml import html
        cells.insert(2, html.th('Debug Info', class_='sortable')) # Add header
    except ImportError:
        logging.warning("py.xml not found, cannot modify pytest-html header.")
    except Exception as e:
        logging.error(f"Error adding pytest-html header: {e}")


@pytest.hookimpl(optionalhook=True)
def pytest_html_results_table_row(report, cells):
    """Add debug info cell content for failed tests."""
    try:
        from py.xml import html
        debug_content = ""
        if report.nodeid in _collected_debug_info and report.failed:
            info = _collected_debug_info[report.nodeid]
            # Format the collected data and analysis hint nicely
            # (This formatting could be much more sophisticated)
            debug_content += "<h4>Collected Data:</h4><pre>"
            import json
            debug_content += json.dumps(info.get("collected", {}), indent=2)
            debug_content += "</pre>"

            if "analysis" in info and info["analysis"].get("ai_hint"):
                debug_content += "<h4>AI Hint:</h4><pre>"
                debug_content += info["analysis"]["ai_hint"]
                debug_content += "</pre>"
            else:
                 debug_content += "<h4>AI Hint:</h4><pre>(Not available or failed)</pre>"

        # Insert the content into the corresponding cell
        cells.insert(2, html.td(html.raw(debug_content), class_='col-debug'))

    except ImportError:
         # Only log warning once if needed
         pass # Fail silently if py.xml not available
    except Exception as e:
        logging.error(f"Error adding pytest-html row content: {e}")
        # Insert placeholder on error
        cells.insert(2, html.td("Error rendering debug info", class_='col-debug'))


# --- Cleanup (Optional) ---
# You might want to clear the global dictionary after the session finishes
# def pytest_sessionfinish(session):
#     _collected_debug_info.clear()
#     logging.info("Cleared collected debug info.")
