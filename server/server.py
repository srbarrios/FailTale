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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout,
                    force=True)
log = logging.getLogger(__name__)

# --- App Initialization ---
app = Flask(__name__)
config = {}  # Global config variable


# --- Helpers ---
def load_environments(config_path):
    """Load environment configuration from a YAML file."""
    global config
    try:
        config = load_config(config_path)
    except Exception as e:
        log.error(f"Failed to load configuration: {e}")
        raise


async def collect_for_host(host_info, ssh_defaults, components_config):
    """Collect useful data for a specific host asynchronously."""
    print(f"Collecting data for host: {host_info}")
    hostname = host_info.get('hostname')
    role = host_info.get('role')
    if not hostname or not role:
        log.warning(f"Skipping invalid host entry: {host_info}")
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
            log.warning(f"Empty command for {hostname}/{role}/{description}")
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


def reduce_html_to_json(page_html):
    """Extracts simplified, visible elements with semantic roles from raw HTML."""
    allowed_tags = { "h1", "h2", "h3", "h4", "h5", "h6",
                    "label", "button", "a", "input", "select",
                    "textarea", "p", "div", "span" }

    error_keywords = ["error", "failed", "exception", "not found", "unauthorized", "invalid", "missing"]
    error_classes = ["text-danger", "has-error", "alert-danger", "alert", "error"]

    soup = BeautifulSoup(page_html, "html.parser")
    result = []

    for tag in soup.find_all(allowed_tags):
        # Filter out non-visible elements by common static rules
        style = tag.get("style", "").lower()
        class_list = tag.get("class", [])
        if any(keyword in style for keyword in ["display:none", "visibility:hidden", "opacity:0"]):
            continue
        if tag.get("aria-hidden") == "true":
            continue
        if "hidden" in tag.get("class",""):
            continue
        if tag.name == "input" and tag.get("type") in {"hidden", "submit"}:
            continue

        # Get visible text or fallback attributes
        text = tag.get_text(separator=" ", strip=True).lower()
        if not text and tag.name == "input":
            text = tag.get("value") or tag.get("placeholder") or tag.get("title")

        if text:
            if any(kwd in text for kwd in error_keywords) or any(any(err in cls.lower() for err in error_classes) for cls in class_list):
                result.append({'possible_error': text})
            else:
                result.append({tag.name: text})

    return result


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
        log.error("Invalid or missing 'hosts' field in request")
        return jsonify({"error": "Missing or invalid 'hosts' field"}), 400
    if not isinstance(test_report, str):
        log.error(f"Invalid 'test_report' format: {type(test_report)}")
        return jsonify({"error": "Invalid 'test_report' format"}), 400

    if not config:
        log.error("Server configuration not loaded")
        return jsonify({"error": "Server configuration not loaded"}), 500

    ssh_defaults = config.get('ssh_defaults', {})
    components_config = config.get('components', {})

    target_hosts = get_hosts_to_collect(all_hosts, test_report, config.get('ollama'))
    if not target_hosts:
        log.error("No hosts identified for collection")
        return jsonify({"error": "Unexpected server error"}), 500

    log.info("Identified hosts for collection: %s", target_hosts)

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
        log.exception(f"Critical error during collection: {e}")
        return jsonify({"error": "Unexpected server error"}), 500


@app.route('/v1/analyze', methods=['POST'])
def analyze_data():
    """Endpoint to analyze collected data and provide root cause hints using LLM."""
    if not request.is_json:
        log.error("Expected JSON request")
        return jsonify({"error": "Expected JSON request"}), 400

    data = request.get_json()
    collected_data = data.get('collected_data')
    page_html = data.get('page_html', 'Unknown page HTML')
    test_report = data.get('test_report', 'Unknown test report')
    test_failure = data.get('test_failure', 'Unknown test failure')

    if not collected_data:
        log.error("Missing 'collected_data' field in request")
        return jsonify({"error": "Missing 'collected_data' field"}), 400
    if not config or not config.get('ollama'):
        log.error("Ollama configuration missing in server configuration")
        return jsonify({"error": "Ollama configuration missing"}), 500
    if not isinstance(collected_data, dict):
        log.error(f"Invalid 'collected_data' format: {type(collected_data)}")
        return jsonify({"error": "Invalid 'collected_data' format"}), 400

    try:
        # Format collected data for LLM prompt
        context_str = ""
        for host, info in collected_data.items():
            context_str += f"--- Host: {host} (Role: {info.get('role', 'unknown')}) ---\n"
            for item in info.get('data', []):
                context_str += f"Log Description: {item.get('description')}\n"
                if item.get('output'):
                    context_str += f"Log Output:\n{item['output'][:500]}\n"
                if item.get('error'):
                    context_str += f"Log Error:\n{item['error'][:500]}\n"
            context_str += "\n"
    except Exception as e:
        log.exception(f"Error formatting collected data for LLM prompt: {e}")
        return jsonify({"error": "Error formatting collected data for LLM prompt"}), 500

    try:
        # Pre-process HTML page to reduce tokens
        if page_html and isinstance(page_html, str):
            reduced_html_json = reduce_html_to_json(page_html)
            reduced_html = json.dumps(reduced_html_json)
        else:
            reduced_html = "Unknown page HTML."
    except Exception as e:
        log.exception(f"Error processing HTML page: {e}")
        return jsonify({"error": "Error processing HTML page"}), 500

    hint = get_root_cause_hint(context_str, reduced_html, test_report, test_failure, config.get('ollama'),
                               with_rag=False)
    return jsonify({"root_cause_hint": hint})
