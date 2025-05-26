# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import logging
import re
import sys

import requests

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout,
                    force=True)
log = logging.getLogger(__name__)

# --- Configuration ---
NUM_CHUNKS_TO_RETRIEVE = 5
TEMPERATURE = 0.5
TOP_K = 20


# --- End Configuration ---

def _validate_ollama_config(ollama_config):
    if not ollama_config or not ollama_config.get("base_url"):
        log.error("Ollama configuration missing or incomplete.")
        return False
    return True


def get_root_cause_hint(context_collected, page_html, test_report, test_failure, ollama_config, with_rag=False):
    """Send collected debug data to Ollama and request a root cause hint."""
    try:
        if with_rag:
            rag_chain = initialize_rag_system(ollama_config)
            ai_response = get_rag_response_with_ollama(
                rag_chain, context_collected, page_html, test_report, test_failure, ollama_config
            )
        else:
            ai_response = get_ollama_root_cause_hint(
                test_report, test_failure, context_collected, page_html, ollama_config
            )
        return ai_response.strip() if ai_response else "No response content received."
    except Exception as e:
        log.exception("Unexpected error while requesting root cause hint: %s", e)
        return f"Unexpected error during AI analysis: {e}"


def get_hosts_to_collect(hosts, test_report, ollama_config):
    """Send a test report to Ollama and get a list of hosts to collect data from."""
    if not _validate_ollama_config(ollama_config):
        return "Error: Ollama configuration is not available."

    ollama_url = f"{ollama_config['base_url'].rstrip('/')}/api/chat"
    model = ollama_config.get("model", "mistral")
    timeout = ollama_config.get("request_timeout", 400)

    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"You are a test automation assistant.\n"
                        f"Available hosts: {json.dumps(hosts)}\n\n"
                        f"Test report: {test_report}\n\n"
                        f"Return a comma-separated list of **up to 3 hostnames** from the available hosts.\n"
                        f"Choose those that most likely match words from the test report.\n"
                        f"Respond ONLY with hostnames separated by commas. No explanation or extra formatting."
                    ),
                },
            ],
            "max_tokens": 20,
            "temperature": TEMPERATURE,
            "stream": False
        }

        log.info("Sending request to Ollama (%s):\n%s", ollama_url, payload)
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        log.info(f"Ollama response:\n%s", response.json())

        if response.status_code != 200:
            log.error("Ollama returned an error: %s", response.content)
            return None

        full_message = json.loads(response.content).get("message", {}).get("content", "").strip()
        hostnames = re.findall(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b', full_message)
        targeted_hosts = [
            host for host in hosts if (host["hostname"] in hostnames or host.get("mandatory"))
        ]

        return targeted_hosts

    except requests.exceptions.Timeout as e:
        log.error("Timeout while contacting Ollama at %s (limit: %s) (method: get_hosts_to_collect): %s", ollama_url, timeout, e)
        return None
    except requests.exceptions.RequestException as e:
        log.error("Connection error while contacting Ollama (method: get_hosts_to_collect): %s", e)
        return None
    except Exception as e:
        log.exception("Unexpected error while requesting host list (method: get_hosts_to_collect): %s", e)
        return None


def get_ollama_root_cause_hint(test_report, test_failure, context_collected, page_html, ollama_config):
    """Send a test context with Ollama and get a root cause hint."""
    if not _validate_ollama_config(ollama_config):
        return "Error: Ollama configuration is not available."

    ollama_url = f"{ollama_config['base_url'].rstrip('/')}/api/chat"
    model = ollama_config.get("model", "mistral")
    timeout = ollama_config.get("request_timeout", 900)
    seed = ollama_config.get("seed", None)

    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"You are a QA Analyst. You will analyze data from a failed automated test. "
                        f"Your task is to provide a single, concise hint that identifies the most likely root cause.\n"
                        f"You are given:\n"
                        f"- The test failure message: {test_failure}\n\n"
                        f"- The full Gherkin test report: {test_report}\n\n"
                        f"- System logs from the test environment: {context_collected}\n\n"
                        f"- Pre-processed current HTML page in JSON format: {page_html}\n\n"
                        f"Instructions:\n"
                        f"- Focus on 'possible_error' keys in the HTML.\n"
                        f"- Look for exact word matches between the failure message and other data.\n"
                        f"- Prioritize logs with the keywords 'error', 'err', or error codes.\n"
                        f"- Don't output: introduction, summary, paths, hypothesis, instructions.\n\n"
                        f"- Output only the final hint.\n\n"
                        f"Hint:"
                    ),
                },
            ],
            "seed": seed,
            "max_tokens": 42,
            "temperature": TEMPERATURE,
            "stream": False
        }

        log.info("Sending request to Ollama (%s):\n%s", ollama_url, payload)
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        log.info(f"Ollama response:\n%s", response.json())

        contents = []
        for line in response.iter_lines():
            if line:
                obj = json.loads(line.decode("utf-8"))
                content = obj.get("message", {}).get("content", "")
                contents.append(content)

        full_message = "".join(contents)
        return full_message

    except requests.exceptions.Timeout as e:
        log.error("Timeout while contacting Ollama at %s (limit: %s) (method: get_ollama_root_cause_hint): %s", ollama_url, timeout, e)
        return f"Error: Timeout while contacting Ollama (limit: {timeout}s)."
    except requests.exceptions.RequestException as e:
        log.error("Connection error while contacting Ollama (method: get_ollama_root_cause_hint): %s", e)
        return f"Connection error with Ollama: {e}"
    except Exception as e:
        log.exception("Unexpected error during AI analysis: %s", e)
        return f"Unexpected error during AI analysis: {e}"


# TODO: This implementation doesn't work as expected. Please refrain from using it yet.
def initialize_rag_system(ollama_config):
    """Initialize the RAG system with LangChain."""
    if not _validate_ollama_config(ollama_config):
        return "Error: Ollama configuration is not available."

    prompt_template = (
        "You are a QA Analyst. You will analyze data from a failed automated test. "
        "Your task is to provide a single, concise hint that identifies the most likely root cause.\n"
        "You are given:\n"
        "1. Test details:|{additional_context}|\n\n"
        "2. Relevant documentation:|{documentation}|\n\n"
        "Instructions:\n"
        f"- Focus on 'possible_error' keys on the HTML page.\n"
        "- Focus on exact word matches between the test failure and logs.\n"
        "- Focus on exact word matches between the test report and the documentation.\n"
        "- Don't output: introduction, summary, paths, hypothesis, instructions.\n\n"
        "- Give facts (example: Description and Output logs related to the hint)\n"
        "Output only the hint.\n"
    )

    model = ollama_config.get("model", "mistral")
    embedding_model = ollama_config.get("embedding_model", "nomic-embed-text")
    persist_directory = ollama_config.get("persist_directory", "chroma_db")
    seed = ollama_config.get("seed", None)

    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS_TO_RETRIEVE})
    llm = ChatOllama(model=model, temperature=TEMPERATURE, top_k=TOP_K, seed=seed, num_ctx=4096)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        doc_result = "\n\n".join(doc.page_content for doc in docs)
        logging.debug("Formatted documents for LLM: %s", doc_result)
        return doc_result

    rag_chain = (
            {"documentation": retriever | format_docs, "additional_context": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


def get_rag_response_with_ollama(rag_chain, context_collected, page_html, test_report, test_failure, ollama_config):
    """Get a RAG system response and forward it to Ollama for analysis."""
    if not _validate_ollama_config(ollama_config):
        return "Error: Ollama configuration is not available."

    try:
        context = (
            f"- Test failure:\n{test_failure}\n\n"
            f"- Test report:\n{test_report}\n\n"
            f"- System logs:\n{context_collected}"
            f"- Pre-processed current HTML page:\n{page_html}"
        )

        ai_response = rag_chain.invoke(context)

        return ai_response.strip() if ai_response else "No response content received."

    except requests.exceptions.Timeout:
        log.error("Timeout while contacting Ollama.")
        return "Error: Timeout while contacting Ollama (method: get_rag_response_with_ollama)."
    except requests.exceptions.RequestException as e:
        log.error("Connection error while contacting Ollama (method: get_rag_response_with_ollama): %s", e)
        return f"Connection error with Ollama: {e}"
    except Exception as e:
        log.exception("Unexpected error during AI analysis with RAG: %s", e)
        return f"Unexpected error during AI analysis: {e}"
