# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import logging
import re

import requests
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

# --- Configuration ---
NUM_CHUNKS_TO_RETRIEVE = 5
TEMPERATURE = 0.8
TOP_K = 40
# --- End Configuration ---

def _validate_ollama_config(ollama_config):
    if not ollama_config or not ollama_config.get('base_url'):
        logging.error("Ollama configuration missing or incomplete.")
        return False
    return True

def get_root_cause_hint(context_collected, test_report, test_failure, ollama_config, with_rag = False):
    """Send collected debug data to Ollama and request a root cause hint."""
    try:
        if with_rag:
            rag_chain = initialize_rag_system(ollama_config)
            ai_response = get_rag_response_with_ollama(rag_chain, context_collected, test_report, test_failure, ollama_config)
        else:
            ai_response = get_ollama_root_cause_hint(test_report, test_failure, context_collected, ollama_config)
        return ai_response.strip() if ai_response else "No response content received."
    except Exception as e:
        logging.exception("Unexpected error while requesting root cause hint: %s", e)
        return f"Unexpected error during AI analysis: {e}"

def get_hosts_to_collect(hosts, test_report, ollama_config):
    """Send a test report to Ollama and get a list of hosts to collect data from."""
    if not _validate_ollama_config(ollama_config):
        return "Error: Ollama configuration is not available."

    ollama_url = f"{ollama_config['base_url'].rstrip('/')}/api/chat"
    model = ollama_config.get('model', 'llama3')
    timeout = ollama_config.get('request_timeout', 60)

    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"This is the list of all available hosts: {json.dumps(hosts)} | "
                        f"Give me a list of comma-separated hostnames to collect data from, "
                        f"based on this test report:\n{test_report}"
                    ),
                },
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()

        contents = []
        for line in response.iter_lines():
            if line:
                obj = json.loads(line.decode('utf-8'))
                content = obj.get("message", {}).get("content", "")
                contents.append(content)

        full_message = ''.join(contents)
        hostnames = re.findall(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b', full_message)
        targeted_hosts = [
            host for host in hosts if (host["hostname"] in hostnames or host.get("mandatory"))
        ]

        return targeted_hosts

    except requests.exceptions.Timeout:
        logging.error("Timeout while contacting Ollama at %s (limit: %ss)", ollama_url, timeout)
        return f"Error: Timeout while contacting Ollama (limit: {timeout}s)."
    except requests.exceptions.RequestException as e:
        logging.error("Connection error while contacting Ollama: %s", e)
        return f"Connection error with Ollama: {e}"
    except Exception as e:
        logging.exception("Unexpected error while requesting host list: %s", e)
        return f"Unexpected error during AI analysis: {e}"

def get_ollama_root_cause_hint(test_report, test_failure, context_collected, ollama_config):
    """Send a test context with Ollama and get a root cause hint."""
    if not _validate_ollama_config(ollama_config):
        return "Error: Ollama configuration is not available."

    ollama_url = f"{ollama_config['base_url'].rstrip('/')}/api/chat"
    model = ollama_config.get('model', 'llama3')
    timeout = ollama_config.get('request_timeout', 60)
    seed = ollama_config.get('seed', None)

    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"You are a QA Analyst, expert in Gherkin, Selenium, Javascript, XPath and Linux. | "
                        f"This is the test failure to analyze: {test_failure} | "
                        f"The failure happens during this test report: {test_report} | "
                        f"These are the system logs produced at the same time, analyze the Desc and Output values: {context_collected} | "
                        f"Give me the most reasonable root cause and hints, based on all the data. Be concise and factual."
                    ),
                },
            ],
            "temperature": 0.6,
            "max_tokens": -1,
            "seed": seed
        }

        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        contents = []
        for line in response.iter_lines():
            if line:
                obj = json.loads(line.decode('utf-8'))
                content = obj.get("message", {}).get("content", "")
                contents.append(content)

        full_message = ''.join(contents)
        return full_message

    except requests.exceptions.Timeout:
        logging.error("Timeout while contacting Ollama at %s (limit: %ss)", ollama_url, timeout)
        return f"Error: Timeout while contacting Ollama (limit: {timeout}s)."
    except requests.exceptions.RequestException as e:
        logging.error("Connection error while contacting Ollama: %s", e)
        return f"Connection error with Ollama: {e}"
    except Exception as e:
        logging.exception("Unexpected error while requesting host list: %s", e)
        return f"Unexpected error during AI analysis: {e}"

def initialize_rag_system(ollama_config):
    # TODO: Refine the prompt, the replies seems to hallucinate due to the RAG
    #       and the documentation provided.
    prompt_template = """
    You are an expert QA analyst.
    
    See all the context of a test failure:
    {additional_context}
    
    Additionally, see here the documentation reference:
    --- BEGIN documentation ---
    {documentation}
    --- END documentation ---
    
    Use ALL the previous data to analyze the test failure and identify the most likely root cause of the failure.
    Be concise and factual.
    """

    """Initialize the RAG system with LangChain."""
    if not _validate_ollama_config(ollama_config):
        return "Error: Ollama configuration is not available."

    model = ollama_config.get('model', 'llama3')
    embedding_model = ollama_config.get('embedding_model', 'nomic-embed-text')
    persist_directory = ollama_config.get('persist_directory', 'chroma_db')
    seed = ollama_config.get('seed', None)
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS_TO_RETRIEVE})
    llm = ChatOllama(model=model, temperature=TEMPERATURE, top_k=TOP_K, seed=seed)
    prompt = ChatPromptTemplate.from_template(prompt_template) # Single message assumed to be from the human

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"documentation": retriever | format_docs, "additional_context": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain

def get_rag_response_with_ollama(rag_chain, context_collected, test_report, test_failure, ollama_config):
    """Get a RAG system response and forward it to Ollama for analysis."""
    if not _validate_ollama_config(ollama_config):
        return "Error: Ollama configuration is not available."

    try:
        context = f"""
        --- BEGIN test_failure ---
        {test_failure}
        --- END test_failure ---
        
        --- BEGIN test_report ---
        {test_report}
        --- END test_report ---

        --- BEGIN context_collected ---
        {context_collected}
        --- END context_collected ---
        
        """
        ai_response = rag_chain.invoke(context)
        return ai_response.strip() if ai_response else "No response content received."

    except requests.exceptions.Timeout:
        logging.error("Timeout while contacting Ollama.")
        return f"Error: Timeout while contacting Ollama."
    except requests.exceptions.RequestException as e:
        logging.error("Connection error while contacting Ollama: %s", e)
        return f"Connection error with Ollama: {e}"
    except Exception as e:
        logging.exception("Unexpected error while processing Ollama response: %s", e)
        return f"Unexpected error during AI analysis: {e}"
