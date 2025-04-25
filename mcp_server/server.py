import asyncio
import logging

from flask import Flask, request, jsonify

from .config_loader import load_config
from .llm_interaction import get_root_cause_hint  # Make sure this is implemented
from .ssh_executor import execute_remote_command_async  # Make sure this is implemented

# from .mcp_protocols import McpRequest, McpResponse # (Placeholder) Import MCP classes/models

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load configuration from YAML file
config = None
def load_environments(config_path):
    global config
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise

# --- Endpoints ---

@app.route('/')
def index():
    """ Basic endpoint to verify the server is running. """
    return jsonify({"message": "MCP Server PoC is running!"})

# -----------------------------------------------------------------------------
# --- IMPORTANT ---
# The following endpoints are **CONCEPTUAL EXAMPLES**.
# You should replace or adapt them according to the **REAL MCP SPECIFICATION**.
# -----------------------------------------------------------------------------

@app.route('/mcp/v1/collect', methods=['POST'])
def collect_data():
    """
    Example endpoint to request data collection.
    Expects a JSON like:
    {
        "hosts": [
            {"hostname": "server.example.com", "role": "server"},
            {"hostname": "proxy.example.com", "role": "proxy"}
        ],
        "context": { "test_id": "test_abc", "failure_reason": "AssertionError" } // Optional
    }
    """
    if not request.is_json:
        return jsonify({"error": "Expected JSON request"}), 400

    data = request.get_json()
    target_hosts = data.get('hosts')
    if not target_hosts or not isinstance(target_hosts, list):
        return jsonify({"error": "Missing or invalid 'hosts' field"}), 400
    if not config:
        return jsonify({"error": "Server configuration not loaded"}), 500

    ssh_user = config.get('ssh_defaults', {}).get('username')
    ssh_key = config.get('ssh_defaults', {}).get('private_key_path')
    ssh_passphrase = config.get('ssh_defaults', {}).get('passphrase')
    ssh_password = config.get('ssh_defaults', {}).get('password')

    async def collect_for_host(host_info):
        hostname = host_info.get('hostname')
        role = host_info.get('role')
        if not hostname or not role:
            logging.warning(f"Skipping invalid host entry: {host_info}")
            return hostname or "unknown", {"role": role, "error": "Invalid host info"}

        component_config = config.get('components', {}).get(role, {}).get('useful_data', [])
        host_results = []

        for item in component_config:
            command = item.get('command')
            description = item.get('description', 'No description')

            if not command:
                logging.warning(f"Empty command for {hostname}/{role}/{description}")
                continue

            current_user = host_info.get('ssh_username', ssh_user)
            current_key = host_info.get('ssh_private_key_path', ssh_key)
            current_passphrase = host_info.get('ssh_passphrase', ssh_passphrase)
            current_password = host_info.get('ssh_password', ssh_password)

            stdout, stderr, status = await execute_remote_command_async(
                hostname=hostname,
                username=current_user,
                command=command,
                private_key_path=current_key,
                passphrase=current_passphrase,
                password=current_password
            )

            host_results.append({
                "description": description,
                "command": command,
                "status": status,
                "output": stdout,
                "error": stderr,
            })

        return hostname, {"role": role, "data": host_results}

    # TODO: Call ollama to identify the hosts to collect from, and collect the data through ollama using the MCP server
    async def run_all():
        tasks = [collect_for_host(host) for host in target_hosts]
        results = await asyncio.gather(*tasks)
        return dict(results)

    try:
        all_results = asyncio.run(run_all())
        return jsonify(all_results)
    except Exception as e:
        logging.exception(f"Critical error during collection: {e}")
        return jsonify({"error": "Unexpected server error"}), 500


@app.route('/mcp/v1/analyze', methods=['POST'])
def analyze_data():
    """
    Example endpoint to request data analysis using an LLM.
    Expects a JSON like:
    {
        "collected_data": { ... previously collected data ... },
        "test_report": "Test X failed due to timeout"
    }
    """
    if not request.is_json:
        return jsonify({"error": "Expected a JSON request"}), 400

    data = request.get_json()
    collected_data = data.get('collected_data')
    test_report = data.get('test_report', 'Unknown error')

    if not collected_data:
        return jsonify({"error": "Missing 'collected_data' field"}), 400
    if not config or not config.get('ollama'):
        return jsonify({"error": "Ollama configuration is not defined in config.yaml"}), 500

    # Format collected data for the LLM prompt
    # (This logic could live in llm_interaction.py)
    context_str = ""
    for host, info in collected_data.items():
        context_str += f"--- Host: {host} (Role: {info.get('role', 'unknown')}) ---\n"
        for item in info.get('data', []):
            context_str += f"Desc: {item.get('description')}\nCmd: {item.get('command')}\nStatus: {item.get('status')}\n"
            if item.get('output'):
                context_str += f"Output:\n{item['output'][:500]}...\n"  # Limit length
            if item.get('error'):
                context_str += f"Error:\n{item['error'][:500]}...\n"   # Limit length
        context_str += "\n"

    hint = get_root_cause_hint(context_str, test_report, config.get('ollama'))  # Pass Ollama config

    # Return the response (adjust format according to MCP)
    return jsonify({"root_cause_hint": hint})
