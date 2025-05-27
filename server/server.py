# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import asyncio
import json
import logging
import sys

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

from .config_loader import load_config
from .llm_interaction import get_root_cause_hint, get_hosts_to_collect
from .ssh_executor import execute_remote_command_async

# --- Logging Configuration ---
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        stream=sys.stdout,
        force=True
    )

# --- App Initialization ---
app = Flask(__name__)
config = {}


# --- Helpers ---
def load_environments(config_path: str):
    """Load environment configuration from a YAML file."""
    log = logging.getLogger(f"{__name__}.load_environments")
    global config
    try:
        loaded_config = load_config(config_path)
        if loaded_config is not None:
            config = loaded_config
            log.info(f"Configuration loaded successfully from {config_path}")
        else:
            log.error(f"Configuration file {config_path} loaded as None. Server might not function correctly.")
            config = {}
    except Exception as e:
        log.exception(f"Failed to load configuration from {config_path}: {e}")
        config = {}
        raise

async def collect_for_host(host_info: dict, ssh_defaults: dict, components_config: dict):
    """Collect useful data for a specific host asynchronously."""
    log = logging.getLogger(f"{__name__}.collect_for_host")

    if not isinstance(host_info, dict):
        log.error(f"Received invalid host_info (type: {type(host_info)}): {host_info}")
        return "invalid_host_info_type", {"role": "unknown", "error": "Invalid host_info structure received"}

    hostname = host_info.get('hostname')
    role = host_info.get('role')

    if not hostname or not role:
        log.warning(f"Skipping invalid host entry (missing hostname or role): {host_info}")
        return hostname or "unknown_hostname", {"role": role or "unknown_role", "error": "Invalid host info (missing hostname or role)"}

    log.info(f"Collecting data for host: {hostname} (Role: {role})")

    # Extract all relevant SSH parameters, using defaults if not overridden in host_info
    current_user = host_info.get('ssh_username', ssh_defaults.get('username'))
    current_key = host_info.get('ssh_private_key_path', ssh_defaults.get('private_key_path'))
    current_passphrase = host_info.get('ssh_passphrase', ssh_defaults.get('passphrase'))
    current_password = host_info.get('ssh_password', ssh_defaults.get('password'))
    current_port = int(host_info.get('ssh_port', ssh_defaults.get('port', 22)))

    # Get execution parameters from ssh_defaults (as defined in ssh_executor.py)
    conn_timeout = int(ssh_defaults.get('conn_timeout', 10))
    cmd_timeout = int(ssh_defaults.get('cmd_timeout', 30))
    known_hosts = ssh_defaults.get('known_hosts')
    retries = int(ssh_defaults.get('retries', 2))
    retry_delay = float(ssh_defaults.get('retry_delay', 3.0))

    if not current_user or (not current_key and not current_password):
        log.error(f"Missing SSH credentials for host {hostname}. Cannot execute commands.")
        return hostname, {"role": role, "data": [], "error": "Missing SSH credentials"}

    host_results = []
    commands_to_run = components_config.get(role, {}).get('useful_data', [])

    if not commands_to_run:
        log.warning(f"No 'useful_data' commands defined for role '{role}' in config for host {hostname}.")
        return hostname, {"role": role, "data": [], "error": f"No commands defined for role '{role}'"}

    for item in commands_to_run:
        command = item.get('command')
        description = item.get('description', 'No description')

        if not command:
            log.warning(f"Empty command definition for {hostname}/{role}/{description}")
            host_results.append({
                "description": description, "command": None, "status": -100,
                "output": None, "error": "Command missing in config",
            })
            continue
        try:
            log.debug(f"Executing on {hostname}: {command}")
            stdout, stderr, status = await execute_remote_command_async(
                hostname=hostname, username=current_user, command=command,
                private_key_path=current_key, passphrase=current_passphrase, password=current_password,
                port=current_port, conn_timeout=conn_timeout, cmd_timeout=cmd_timeout,
                known_hosts=known_hosts, retries=retries, retry_delay=retry_delay
            )
            host_results.append({
                "description": description, "command": command, "status": status,
                "output": stdout, "error": stderr,
            })
        except Exception as cmd_exc:
            log.exception(f"Error executing command '{description}' on {hostname}: {cmd_exc}")
            host_results.append({
                "description": description, "command": command, "status": -999,
                "output": None, "error": f"Failed to execute command: {cmd_exc}",
            })
    return hostname, {"role": role, "data": host_results}


def reduce_html_to_json(page_html: str) -> list:
    """Extracts simplified, visible elements with semantic roles from raw HTML."""
    log = logging.getLogger(f"{__name__}.reduce_html_to_json")
    if not page_html or not isinstance(page_html, str):
        log.warning("reduce_html_to_json received empty or invalid HTML content.")
        return []

    allowed_tags = { "h1", "h2", "h3", "h4", "h5", "h6",
                     "label", "button", "a", "input", "select",
                     "textarea", "p", "div", "span" }
    error_keywords = ["error", "failed", "exception", "not found", "unauthorized", "invalid", "missing", "warning"]
    error_classes = ["text-danger", "has-error", "alert-danger", "alert", "error", "warning"]

    try:
        soup = BeautifulSoup(page_html, "html.parser")
    except Exception as e:
        log.exception(f"BeautifulSoup failed to parse HTML: {e}")
        return [{"parsing_error": "Failed to parse HTML content"}]

    result = []
    seen_texts = set()

    for tag in soup.find_all(allowed_tags):
        try:
            style = tag.get("style", "").lower()
            class_list = tag.get("class", [])
            if any(keyword in style for keyword in ["display:none", "visibility:hidden", "opacity:0"]):
                continue
            if tag.get("aria-hidden") == "true":
                continue
            if "hidden" in class_list or ("d-none" in class_list):
                continue
            if tag.name == "input" and tag.get("type") in {"hidden", "submit", "reset"}:
                continue

            content = []
            if tag.name == "input":
                input_type = tag.get("type", "text").lower()
                if input_type == "button" or input_type == "submit" or input_type == "reset":
                    content.append(tag.get("value",""))
                else:
                    content.append(tag.get("aria-label", tag.get("placeholder", tag.get("value", ""))))
            else:
                content.append(tag.get_text(separator=" ", strip=True))

            if tag.get("title"): content.append(tag.get("title"))
            if tag.get("aria-label"): content.append(tag.get("aria-label"))

            text = " ".join(filter(None, content)).strip().lower()

            if text and text not in seen_texts:
                seen_texts.add(text)
                is_error_related = any(kwd in text for kwd in error_keywords) or \
                                   any(any(err in cls.lower() for err in error_classes) for cls in class_list)
                if is_error_related:
                    element_info = {"possible_error": text}
                else:
                    element_info = {tag.name: text}
                result.append(element_info)
        except Exception as tag_proc_e:
            log.exception(f"Error processing tag {tag.name}: {tag_proc_e}")
            continue
    log.info(f"Reduced HTML to {len(result)} JSON elements.")
    return result


# --- Endpoints ---
@app.route('/')
def index():
    """Basic endpoint to verify the server is running."""
    log = logging.getLogger(f"{__name__}.index")
    log.info("Index route '/' accessed.")
    return jsonify({"message": "FailTale Server PoC is running!"})


@app.route('/v1/collect', methods=['POST'])
def collect_data():
    """Endpoint to collect data from specified hosts based on test report."""
    log = logging.getLogger(f"{__name__}.collect_data")
    log.info("Received request for /v1/collect")
    if not request.is_json:
        log.warning("Invalid request to /v1/collect: not JSON")
        return jsonify({"error": "Expected JSON request"}), 400

    data = request.get_json()
    log.debug(f"Collect request payload: {json.dumps(data, indent=2)}")
    all_hosts_from_request = data.get('hosts')
    test_report = data.get('test_report')

    if not all_hosts_from_request or not isinstance(all_hosts_from_request, list):
        log.error("Invalid or missing 'hosts' field in /v1/collect request")
        return jsonify({"error": "Missing or invalid 'hosts' field"}), 400
    if not isinstance(test_report, str):
        log.error(f"Invalid or missing 'test_report' in /v1/collect request (type: {type(test_report)})")
        return jsonify({"error": "Missing or invalid 'test_report' field"}), 400

    if not config:
        log.critical("Server configuration not loaded. Cannot process /v1/collect.")
        return jsonify({"error": "Server configuration not loaded"}), 500

    ssh_defaults = config.get('ssh_defaults', {})
    components_config = config.get('components', {})

    llm_provider_name = config.get("default_llm")
    llm_provider_settings = config.get(f"{llm_provider_name}", {})

    if not llm_provider_name or not llm_provider_settings:
        log.error("LLM provider or its settings missing in server configuration for /v1/collect.")
        return jsonify({"error": "LLM configuration missing on server"}), 500

    # --- Determine target hosts using LLM ---
    try:
        log.info(f"Requesting host selection from LLM provider: {llm_provider_name}...")
        # Pass provider_name and provider_config to get_hosts_to_collect
        target_hosts = get_hosts_to_collect(all_hosts_from_request, test_report, llm_provider_name, llm_provider_settings)

        # Log the actual hostnames if target_hosts is a list of dicts
        if isinstance(target_hosts, list):
            log.info(f"Hosts identified by LLM for collection: {[h.get('hostname') for h in target_hosts if isinstance(h, dict)]}")
        else:
            log.warning(f"get_hosts_to_collect returned non-list: {target_hosts}")


        if not isinstance(target_hosts, list):
            log.error(f"get_hosts_to_collect returned invalid type: {type(target_hosts)}. Content: {target_hosts}")
            return jsonify({"error": "LLM failed to return a valid host list"}), 500
        if not target_hosts: # Empty list means no hosts identified
            log.warning("No target hosts identified by LLM or matching criteria. Returning empty result.")
            return jsonify({})

        valid_target_hosts = [h for h in target_hosts if isinstance(h, dict) and h.get("hostname") and h.get("role")]
        if len(valid_target_hosts) != len(target_hosts):
            log.warning("Some invalid entries were filtered from LLM's target_hosts list.")
        if not valid_target_hosts:
            log.warning("No valid target hosts remained after filtering LLM response. Returning empty result.")
            return jsonify({})
    except Exception as e:
        log.exception(f"Error during host selection via LLM: {e}")
        return jsonify({"error": "Failed to select hosts for collection due to LLM interaction error"}), 500

    # --- Define the async runner ---
    async def run_all_collections():
        log.info(f"Starting parallel collection for {len(valid_target_hosts)} hosts.")
        tasks = [
            collect_for_host(host, ssh_defaults, components_config)
            for host in valid_target_hosts
        ]
        results_list_of_tuples = await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Finished parallel collection tasks.")

        final_results = {}
        for i, result_item in enumerate(results_list_of_tuples):
            original_host_info = valid_target_hosts[i]
            if isinstance(result_item, Exception):
                log.error(f"Task for host '{original_host_info.get('hostname')}' failed with exception: {result_item}")
                final_results[original_host_info.get('hostname', f"unknown_error_host_{i}")] = {
                    "role": original_host_info.get('role', 'unknown'),
                    "error": f"Collection task failed: {result_item}"
                }
            elif isinstance(result_item, tuple) and len(result_item) == 2:
                hostname, host_data = result_item
                final_results[hostname] = host_data
            else:
                log.error(f"Unexpected result type from gather task for host '{original_host_info.get('hostname')}': {result_item}")
                final_results[original_host_info.get('hostname', f"unknown_gather_result_{i}")] = {
                    "role": original_host_info.get('role', 'unknown'),
                    "error": "Invalid result structure from collection task"
                }
        return final_results

    # --- Run the async code ---
    try:
        all_results = asyncio.run(run_all_collections())
        log.info("Collection processing completed.")
        return jsonify(all_results)
    except RuntimeError as e:
        log.error(f"RuntimeError running asyncio tasks (possibly nested loops): {e}")
        return jsonify({"error": "Server concurrency error during collection"}), 500
    except Exception as e:
        log.exception(f"Critical error during collection processing: {e}")
        return jsonify({"error": "Unexpected server error during collection"}), 500


@app.route('/v1/analyze', methods=['POST'])
def analyze_data():
    """Endpoint to analyze collected data and provide root cause hints using LLM."""
    log = logging.getLogger(f"{__name__}.analyze_data")
    log.info("Received request for /v1/analyze")
    if not request.is_json:
        log.warning("Invalid request to /v1/analyze: not JSON")
        return jsonify({"error": "Expected JSON request"}), 400

    data = request.get_json()
    log.debug(f"Analyze request payload: {json.dumps(data, indent=2)}")
    collected_data = data.get('collected_data')
    page_html = data.get('page_html', '')
    test_report = data.get('test_report', 'Unknown test report')
    test_failure = data.get('test_failure', 'Unknown test failure')

    if not collected_data:
        log.warning("Invalid request to /v1/analyze: missing 'collected_data'")
        return jsonify({"error": "Missing 'collected_data' field"}), 400
    if not isinstance(collected_data, dict):
        log.error(f"Invalid 'collected_data' format in /v1/analyze: expected dict, got {type(collected_data)}")
        return jsonify({"error": "Invalid 'collected_data' format, must be an object"}), 400

    if not config:
        log.critical("Server configuration not loaded. Cannot process /v1/analyze.")
        return jsonify({"error": "Server configuration not loaded"}), 500

    # --- Get LLM provider name and its specific settings ---
    llm_provider_name = config.get("default_llm")
    llm_provider_settings = config.get(f"{llm_provider_name}", {})

    if not llm_provider_name or not llm_provider_settings:
        log.error("LLM provider or its settings missing in server configuration for /v1/analyze.")
        return jsonify({"error": "LLM configuration missing on server"}), 500

    # --- Format collected data for LLM prompt ---
    context_str = ""
    try:
        for host, info in collected_data.items():
            if not isinstance(info, dict):
                log.warning(f"Skipping invalid data format for host {host} in collected_data (analyze).")
                continue
            context_str += f"--- Host: {host} (Role: {info.get('role', 'unknown')}) ---\n"
            host_data_list = info.get('data', [])
            if isinstance(host_data_list, list):
                for item in host_data_list:
                    if isinstance(item, dict):
                        context_str += f"Log Description: {item.get('description', 'N/A')}\n"
                        command_executed = item.get('command')
                        if command_executed: context_str += f"Command Executed: {command_executed}\n"
                        context_str += f"Status Code: {item.get('status', 'N/A')}\n"
                        output = item.get('output')
                        if output: context_str += f"Output:\n{str(output)[:1000]}...\n"
                        error = item.get('error')
                        if error: context_str += f"Error Output:\n{str(error)[:1000]}...\n"
                    else:
                        log.warning(f"Skipping non-dict item in data list for host {host} (analyze): {item}")
            else:
                log.warning(f"Invalid 'data' field (not a list) for host {host} (analyze): {host_data_list}")
            host_error = info.get('error')
            if host_error: context_str += f"Host-Level Collection Error: {host_error}\n"
            context_str += "\n"
    except Exception as format_exc:
        log.exception(f"Error formatting collected data for LLM prompt: {format_exc}")
        return jsonify({"error": "Internal server error formatting data"}), 500

    # --- Pre-process HTML page ---
    reduced_html_str = "No HTML page data provided or processed."
    try:
        if page_html and isinstance(page_html, str):
            log.info(f"Processing HTML page content (length: {len(page_html)} chars)...")
            reduced_html_json = reduce_html_to_json(page_html)
            reduced_html_str = json.dumps(reduced_html_json)
            log.info(f"Reduced HTML to JSON (length: {len(reduced_html_str)} chars).")
        elif page_html:
            log.warning(f"page_html provided but is not a string (type: {type(page_html)}). Skipping HTML processing.")
    except Exception as e:
        log.exception(f"Error processing HTML page: {e}")
        reduced_html_str = "Error processing HTML page."

    # --- Get Hint from LLM ---
    try:
        log.info(f"Requesting analysis from LLM provider: {llm_provider_name}...")
        # Pass provider_name and its specific config
        hint = get_root_cause_hint(
            context_collected=context_str,
            page_html=reduced_html_str,
            test_report=test_report,
            test_failure=test_failure,
            provider_name=llm_provider_name,
            provider_config=llm_provider_settings,
            with_rag=config.get('use_rag', False) # Example: make RAG usage configurable from main config
        )
        log.info("Analysis received from LLM.")
        return jsonify({"root_cause_hint": hint})
    except Exception as llm_exc:
        log.exception(f"Error during LLM interaction: {llm_exc}")
        return jsonify({"error": "Failed to get analysis from LLM"}), 500
