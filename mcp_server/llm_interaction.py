import logging

import requests


def get_ai_hint(context_data_str, failure_summary, ollama_config):
    """Sends collected debug data to Ollama and requests a root cause hint."""

    if not ollama_config or not ollama_config.get('base_url'):
        logging.error("Ollama configuration missing or incomplete.")
        return "Error: Ollama configuration is not available."

    ollama_url = f"{ollama_config['base_url'].rstrip('/')}/api/chat"
    model = ollama_config.get('model', 'llama3')
    timeout = ollama_config.get('request_timeout', 60)

    prompt = f"""
    Analyze the following test failure context.
    Failure Summary: {failure_summary}

    Collected Debug Data:
    --- BEGIN DATA ---
    {context_data_str}
    --- END DATA ---

    Based **only** on this data, identify the most likely root cause or 2–3 key clues.
    Be concise. If the data is inconclusive, say so clearly. Do not make up information.
    Response:
    """

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False  # Request a complete response
    }

    try:
        logging.info(f"Sending request to Ollama at {ollama_url} using model '{model}'")
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()

        response_data = response.json()
        ai_response = response_data.get('message', {}).get('content', '')

        logging.info("Response received from Ollama.")
        return ai_response.strip() if ai_response else "No response content received."

    except requests.exceptions.Timeout:
        logging.error(f"Timeout while contacting Ollama at {ollama_url} (limit: {timeout}s)")
        return f"Error: Timeout while contacting Ollama (limit: {timeout}s)."
    except requests.exceptions.RequestException as e:
        logging.error(f"Connection error while contacting Ollama: {e}")
        return f"Connection error with Ollama: {e}"
    except Exception as e:
        logging.exception(f"Unexpected error while processing Ollama response: {e}")
        return f"Unexpected error during AI analysis: {e}"
