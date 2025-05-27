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
    conn_timeout = int(ssh_defaults.get('conn_timeout', 10))
    cmd_timeout = int(ssh_defaults.get('cmd_timeout', 180))
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
            host_results.append({"description": description, "command": None, "status": -100, "output": None, "error": "Command missing in config"})
            continue
        try:
            log.debug(f"Executing on {hostname}: {command}")
            stdout, stderr, status = await execute_remote_command_async(
                hostname=hostname, username=current_user, command=command,
                private_key_path=current_key, passphrase=current_passphrase, password=current_password,
                port=current_port, conn_timeout=conn_timeout, cmd_timeout=cmd_timeout,
                known_hosts=known_hosts, retries=retries, retry_delay=retry_delay
            )
            host_results.append({"description": description, "command": command, "status": status, "output": stdout, "error": stderr})
        except Exception as cmd_exc:
            log.exception(f"Error executing command '{description}' on {hostname}: {cmd_exc}")
            host_results.append({"description": description, "command": command, "status": -999, "output": None, "error": f"Failed to execute command: {cmd_exc}"})
    return hostname, {"role": role, "data": host_results}

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

    llm_provider_name = config.get("default_llm_provider")
    llm_provider_settings = config.get(f"{llm_provider_name}", {})

    if not llm_provider_name or not llm_provider_settings:
        log.error("LLM provider or its settings missing in server configuration for /v1/collect.")
        return jsonify({"error": "LLM configuration missing on server"}), 500
    try:
        log.info(f"Requesting host selection from LLM provider: {llm_provider_name}...")
        target_hosts = get_hosts_to_collect(all_hosts_from_request, test_report, llm_provider_name, llm_provider_settings)
        if isinstance(target_hosts, list):
            log.info(f"Hosts identified by LLM for collection: {[h.get('hostname') for h in target_hosts if isinstance(h, dict)]}")
        else:
            log.warning(f"get_hosts_to_collect returned non-list: {target_hosts}")
        if not isinstance(target_hosts, list):
            log.error(f"get_hosts_to_collect returned invalid type: {type(target_hosts)}. Content: {target_hosts}")
            return jsonify({"error": "LLM failed to return a valid host list"}), 500
        if not target_hosts:
            log.warning("No target hosts identified by LLM or matching criteria. Returning empty result.")
            return jsonify({})
        valid_target_hosts = [h for h in target_hosts if isinstance(h, dict) and h.get("hostname") and h.get("role")]
        if len(valid_target_hosts) != len(target_hosts): log.warning("Some invalid entries were filtered from LLM's target_hosts list.")
        if not valid_target_hosts:
            log.warning("No valid target hosts remained after filtering LLM response. Returning empty result.")
            return jsonify({})
    except Exception as e:
        log.exception(f"Error during host selection via LLM: {e}")
        return jsonify({"error": "Failed to select hosts for collection due to LLM interaction error"}), 500

    async def run_all_collections():
        log.info(f"Starting parallel collection for {len(valid_target_hosts)} hosts.")
        tasks = [collect_for_host(host, ssh_defaults, components_config) for host in valid_target_hosts]
        results_list_of_tuples = await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Finished parallel collection tasks.")
        final_results = {}
        for i, result_item in enumerate(results_list_of_tuples):
            original_host_info = valid_target_hosts[i]
            if isinstance(result_item, Exception):
                log.error(f"Task for host '{original_host_info.get('hostname')}' failed with exception: {result_item}")
                final_results[original_host_info.get('hostname', f"unknown_error_host_{i}")] = {"role": original_host_info.get('role', 'unknown'), "error": f"Collection task failed: {result_item}"}
            elif isinstance(result_item, tuple) and len(result_item) == 2:
                hostname, host_data = result_item
                final_results[hostname] = host_data
            else:
                log.error(f"Unexpected result type from gather task for host '{original_host_info.get('hostname')}': {result_item}")
                final_results[original_host_info.get('hostname', f"unknown_gather_result_{i}")] = {"role": original_host_info.get('role', 'unknown'), "error": "Invalid result structure from collection task"}
        return final_results
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
    test_report = data.get('test_report', 'Unknown test report')
    test_failure = data.get('test_failure', 'Unknown test failure')
    screenshot_b64_str = data.get('screenshot', '')

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
    llm_provider_name = config.get("default_llm_provider")
    llm_provider_settings = config.get(f"{llm_provider_name}", {})

    if not llm_provider_name or not llm_provider_settings:
        log.error("LLM provider or its settings missing in server configuration for /v1/analyze.")
        return jsonify({"error": "LLM configuration missing on server"}), 500

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

    try:
        log.info(f"Requesting analysis from LLM provider: {llm_provider_name}...")
        hint = get_root_cause_hint(
            context_collected=context_str,
            screenshot_b64_str=screenshot_b64_str,
            test_report=test_report,
            test_failure=test_failure,
            provider_name=llm_provider_name,
            provider_config=llm_provider_settings,
            with_rag=config.get('use_rag', False)
        )
        log.info("Analysis received from LLM.")
        return jsonify({"root_cause_hint": hint})
    except Exception as llm_exc:
        log.exception(f"Error during LLM interaction: {llm_exc}")
        return jsonify({"error": "Failed to get analysis from LLM"}), 500
