# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import asyncio
import logging

from flask import Flask, request, jsonify

from .config_loader import load_config
from .llm_interaction import get_root_cause_hint, get_hosts_to_collect
from .ssh_executor import execute_remote_command_async

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- App Initialization ---
app = Flask(__name__)
config = None  # Global config variable


# --- Helpers ---
def load_environments(config_path):
    """Load environment configuration from a YAML file."""
    global config
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise


async def collect_for_host(host_info, ssh_defaults, components_config):
    """Collect useful data for a specific host asynchronously."""
    print(f"Collecting data for host: {host_info}")
    hostname = host_info.get('hostname')
    role = host_info.get('role')
    if not hostname or not role:
        logging.warning(f"Skipping invalid host entry: {host_info}")
        return hostname or "unknown", {"role": role, "error": "Invalid host info"}

    current_user = host_info.get('ssh_username', ssh_defaults.get('username'))
    current_key = host_info.get('ssh_private_key_path', ssh_defaults.get('private_key_path'))
    current_passphrase = host_info.get('ssh_passphrase', ssh_defaults.get('passphrase'))
    current_password = host_info.get('ssh_password', ssh_defaults.get('password'))

    host_results = []
    for item in components_config.get(role, {}).get('useful_data', []):
        command = item.get('command')
        description = item.get('description', 'No description')

        if not command:
            logging.warning(f"Empty command for {hostname}/{role}/{description}")
            continue

        stdout, stderr, status = await execute_remote_command_async(
            hostname=hostname,
            username=current_user,
            command=command,
            private_key_path=current_key,
            passphrase=current_passphrase,
            password=current_password,
        )

        host_results.append({
            "description": description,
            "command": command,
            "status": status,
            "output": stdout,
            "error": stderr,
        })

    return hostname, {"role": role, "data": host_results}


# --- Endpoints ---
@app.route('/')
def index():
    """Basic endpoint to verify the server is running."""
    return jsonify({"message": "Server PoC is running!"})


@app.route('/v1/collect', methods=['POST'])
def collect_data():
    """Endpoint to collect data from specified hosts based on test report."""
    if not request.is_json:
        return jsonify({"error": "Expected JSON request"}), 400

    data = request.get_json()
    all_hosts = data.get('hosts')
    test_report = data.get('test_report')

    if not all_hosts or not isinstance(all_hosts, list):
        return jsonify({"error": "Missing or invalid 'hosts' field"}), 400
    if not isinstance(test_report, str):
        return jsonify({"error": "Invalid 'test_report' format"}), 400

    if not config:
        return jsonify({"error": "Server configuration not loaded"}), 500

    ssh_defaults = config.get('ssh_defaults', {})
    components_config = config.get('components', {})

    target_hosts = get_hosts_to_collect(all_hosts, test_report, config.get('ollama'))
    if not target_hosts:
        return jsonify({"error": "Unexpected server error"}), 500

    logging.info("Identified hosts for collection: %s", target_hosts)

    async def run_all_collections():
        tasks = [
            collect_for_host(host, ssh_defaults, components_config)
            for host in target_hosts
        ]
        results = await asyncio.gather(*tasks)
        return dict(results)

    try:
        all_results = asyncio.run(run_all_collections())
        return jsonify(all_results)
    except Exception as e:
        logging.exception(f"Critical error during collection: {e}")
        return jsonify({"error": "Unexpected server error"}), 500


@app.route('/v1/analyze', methods=['POST'])
def analyze_data():
    """Endpoint to analyze collected data and provide root cause hints using LLM."""
    if not request.is_json:
        return jsonify({"error": "Expected JSON request"}), 400

    data = request.get_json()
    collected_data = data.get('collected_data')
    test_report = data.get('test_report', 'Unknown test report')
    test_failure = data.get('test_failure', 'Unknown test failure')

    if not collected_data:
        return jsonify({"error": "Missing 'collected_data' field"}), 400
    if not config or not config.get('ollama'):
        return jsonify({"error": "Ollama configuration missing"}), 500

    # Format collected data for LLM prompt
    context_str = ""
    for host, info in collected_data.items():
        context_str += f"--- Host: {host} (Role: {info.get('role', 'unknown')}) ---\n"
        for item in info.get('data', []):
            context_str += f"Desc: {item.get('description')}\n"
            if item.get('output'):
                context_str += f"Output:\n{item['output'][:500]}\n"
            if item.get('error'):
                context_str += f"Error:\n{item['error'][:500]}\n"
        context_str += "\n"

    hint = get_root_cause_hint(context_str, test_report, test_failure, config.get('ollama'), with_rag=False)
    return jsonify({"root_cause_hint": hint})
