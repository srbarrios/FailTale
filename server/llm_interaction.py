# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import logging
import re
import os
import sys
import requests
import google.generativeai as genai
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Logging Configuration ---
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        stream=sys.stdout,
        force=True
    )
log = logging.getLogger(__name__)

# --- Default Configuration ---
NUM_CHUNKS_TO_RETRIEVE = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = -2
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_SEED = 42
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_NUM_CONTEXT = 2048

# --- Helper to validate LLM config ---
def _validate_llm_provider_config(provider_config: dict, provider_name: str) -> bool:
    if not provider_config:
        log.error(f"{provider_name} configuration settings missing.")
        return False

    if not provider_config.get("model"):
        log.error("Missing 'model'.")
        return False

    if provider_name == "ollama" and not provider_config.get("base_url"):
        log.error("Ollama settings missing 'base_url'.")
        return False

    if provider_name in {"gemini", "openai"}:
        api_key = provider_config.get("api_key")
        api_key_env_var = provider_config.get("api_key_env_var")

        if not api_key and not api_key_env_var:
            log.error(
                f"{provider_name.capitalize()} API key not set in config 'api_key' or via environment variable 'api_key_env_var'."
            )
            return False

        if api_key_env_var and not os.getenv(api_key_env_var):
            log.error(f"{provider_name.capitalize()} API key environment variable '{api_key_env_var}' is not set.")
            return False

    return True

# --- Unified LLM Interaction Function ---
def _call_llm_api(messages: list, provider_name: str, provider_config: dict) -> str:
    """
    Calls the specified LLM provider API.
    """

    if provider_name == "ollama":
        if not _validate_llm_provider_config(provider_config, "ollama"):
            return "Error: Invalid Ollama configuration."

        ollama_url = f"{provider_config['base_url'].rstrip('/')}/api/chat"
        model = provider_config.get("model", "mistral")
        timeout = int(provider_config.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))

        options = {
            "temperature": provider_config.get("temperature", DEFAULT_TEMPERATURE),
            "num_ctx": provider_config.get("num_ctx", DEFAULT_NUM_CONTEXT),
            "seed": provider_config.get("seed", DEFAULT_SEED),
            "top_k": provider_config.get("top_k"),
            "max_tokens": provider_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        }
        options = {k: v for k, v in options.items() if v is not None}

        payload = {"model": model, "messages": messages, "options": options, "stream": False}
        log.debug(f"Ollama API Request to {ollama_url} with model {model}: {json.dumps(payload, indent=2)}")
        try:
            response = requests.post(ollama_url, json=payload, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            log.debug(f"Ollama API Response: {json.dumps(response_data, indent=2)}")
            return response_data.get("message", {}).get("content", "")
        except requests.exceptions.Timeout:
            log.error(f"Timeout contacting Ollama at {ollama_url} (limit: {timeout}s)")
            return f"Error: Timeout contacting Ollama."
        except requests.exceptions.RequestException as e:
            log.error(f"Connection error contacting Ollama: {e}")
            return f"Error: Connection error with Ollama: {e}"
        except Exception as e:
            log.exception(f"Unexpected error with Ollama API call: {e}")
            return f"Error: Unexpected error with Ollama: {e}"

    elif provider_name == "gemini":
        if not _validate_llm_provider_config(provider_config, "gemini"):
            return "Error: Invalid Gemini configuration."

        api_key = provider_config.get("api_key")
        if not api_key:
            api_key = os.getenv(provider_config.get("api_key_env_var"))

        model_name = provider_config.get("model")

        try:
            genai.configure(api_key=api_key)
            prompt_content = ""
            if messages:
                full_prompt_parts = []
                for msg in messages:
                    if msg.get("role") == "system":
                        full_prompt_parts.append(f"System Instructions: {msg['content']}")
                    elif msg.get("role") == "user":
                        full_prompt_parts.append(msg['content'])
                prompt_content = "\n\n".join(full_prompt_parts)

            if not prompt_content:
                log.warning("No content derived from messages for Gemini prompt.")
                return "Error: No prompt content for Gemini."

            generation_config_params = {
                "temperature": provider_config.get("temperature", DEFAULT_TEMPERATURE),
                "top_k": provider_config.get("top_k"),
                "max_output_tokens": provider_config.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
            }
            generation_config = genai.types.GenerationConfig(
                **{k:v for k,v in generation_config_params.items() if v is not None}
            )

            model = genai.GenerativeModel(model_name)
            log.debug(f"Gemini API Request: Model='{model_name}', Config={generation_config_params}, Content='{prompt_content}'")
            response = model.generate_content(prompt_content, generation_config=generation_config)
            log.debug(f"Gemini API Raw Response: {response}")

            if not response.candidates or not hasattr(response, 'text') or not response.text:
                log.error(f"Gemini API: No candidates or text returned. Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                finish_reason = response.prompt_feedback.block_reason if hasattr(response, 'prompt_feedback') and response.prompt_feedback else "Unknown"
                return f"Error: Gemini API - No content generated. Reason: {finish_reason}"
            return response.text
        except Exception as e:
            log.exception(f"Unexpected error with Gemini API call: {e}")
            return f"Error: Unexpected error with Gemini: {e}"

    elif provider_name == "openai":
        if not _validate_llm_provider_config(provider_config, "openai"):
            return "Error: Invalid OpenAI configuration."

        api_key = provider_config.get("api_key")
        if not api_key:
            api_key = os.getenv(provider_config.get("api_key_env_var"))

        client = OpenAI(api_key=api_key)
        model_name = provider_config.get("model")
        timeout = float(provider_config.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))

        try:
            openai_messages = []
            for msg in messages:
                role = msg.get("role")
                if role not in ["system", "user", "assistant"]:
                    log.warning(f"Mapping unknown role '{role}' to 'user' for OpenAI.")
                    role = "user"
                openai_messages.append({"role": role, "content": msg.get("content", "")})

            if not openai_messages:
                log.warning("No messages to send to OpenAI.")
                return "Error: No messages for OpenAI."

            completion_params = {
                "model": model_name,
                "messages": openai_messages,
                "temperature": provider_config.get("temperature", DEFAULT_TEMPERATURE),
                "top_p": provider_config.get("top_p"),
                "max_tokens": provider_config.get("max_tokens", DEFAULT_MAX_TOKENS)
            }
            completion_params = {k: v for k, v in completion_params.items() if v is not None}

            log.debug(f"OpenAI API Request: Model='{model_name}', Params={completion_params}, Messages='{json.dumps(openai_messages, indent=2)}'")
            response = client.chat.completions.create(
                **completion_params,
                timeout=timeout
            )
            log.debug(f"OpenAI API Raw Response: {response}")
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                log.error(f"OpenAI API: No choices or message content in response. Finish reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")
                return "Error: OpenAI API - No content in response."
        except Exception as e:
            log.exception(f"Unexpected error with OpenAI API call: {e}")
            return f"Error: Unexpected error with OpenAI: {e}"
    else:
        log.error(f"Unsupported LLM provider: {provider_name}")
        return "Error: Unsupported LLM provider."

def _get_llm_root_cause_hint_direct_api(test_report:str, test_failure:str, context_collected:str, page_html:str, provider_name: str, provider_config: dict) -> str:

    if not provider_config:
        log.error(f"Configuration for provider '{provider_name}' not found for root cause hint.")
        return "Error: LLM provider configuration missing."

    prompt_content = (
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
        f"- Don't output: introduction, summary or instructions.\n\n"
        f"- Give facts (example: Description and Output logs related to the hint)\n"
        f"Hint:"
    )
    messages = [{"role": "user", "content": prompt_content}]
    return _call_llm_api(messages, provider_name, provider_config)

# --- Public Functions ---

def get_hosts_to_collect(hosts: list, test_report: str, provider_name: str, provider_config: dict) -> list:
    if not provider_config:
        log.error(f"Configuration for provider '{provider_name}' not found.")
        return []

    prompt_content = (
        f"You are a test automation assistant.\n"
        f"Available hosts: {json.dumps(hosts)}\n\n"
        f"Test report: {test_report}\n\n"
        f"Return a comma-separated list of **up to 3 hostnames** from the available hosts.\n"
        f"Choose those that most likely match words from the test report.\n"
        f"Respond ONLY with hostnames separated by commas. No explanation or extra formatting."
    )
    messages = [{"role": "user", "content": prompt_content}]
    full_message = _call_llm_api(messages, provider_name, provider_config)

    if full_message is None or full_message.startswith("Error:"):
        log.error(f"Failed to get host list from LLM ({provider_name}): {full_message}")
        return []

    log.debug(f"LLM Response for host selection ({provider_name}): {full_message}")
    hostnames_found = re.findall(r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b', full_message)
    hostnames_found.extend(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', full_message))
    hostnames_found = list(set(name.strip().lower() for name in hostnames_found if name.strip()))
    log.info(f"Hostnames/IPs extracted from LLM response: {hostnames_found}")

    targeted_hosts = [
        host for host in hosts if isinstance(host, dict) and (
                (host.get("hostname", "").lower() in hostnames_found) or host.get("mandatory")
        )
    ]
    log.info(f"Final targeted hosts: {[h.get('hostname') for h in targeted_hosts]}")
    return targeted_hosts

def get_root_cause_hint(context_collected:str, page_html:str, test_report:str, test_failure:str, provider_name: str, provider_config: dict, with_rag:bool=False) -> str:
    if not provider_config:
        log.error(f"Configuration for LLM provider '{provider_name}' not found.")
        return "Error: LLM provider configuration missing."

    try:
        if with_rag:
            log.info(f"Attempting RAG-based hint with provider: {provider_name}")
            # Ensure API keys are configured if needed by the provider for RAG
            # This logic will now prioritize config, then env var for RAG initialization as well
            api_key_for_rag = provider_config.get("api_key")
            if not api_key_for_rag:
                api_key_env_var_name = "OPENAI_API_KEY" # Default for OpenAI Langchain
                if provider_name == "gemini":
                    api_key_env_var_name = provider_config.get("api_key_env_var", "GOOGLE_API_KEY")
                elif provider_name == "openai" and "api_key_env_var" in provider_config:
                    api_key_env_var_name = provider_config["api_key_env_var"]

                api_key_for_rag = os.getenv(api_key_env_var_name)

            if provider_name in ["gemini", "openai"] and not api_key_for_rag:
                log.error(f"{provider_name.capitalize()} API key not found in config or environment for RAG.")
                return f"Error: {provider_name.capitalize()} API key not set for RAG."

            # Configure SDKs if API key is sourced for Langchain components
            if provider_name == "gemini" and api_key_for_rag:
                genai.configure(api_key=api_key_for_rag)
            # For OpenAI, Langchain components usually pick up env var or can be passed api_key

            rag_chain = initialize_rag_system(provider_name, provider_config, api_key_for_rag_if_needed=api_key_for_rag)
            ai_response = get_rag_response(
                rag_chain, context_collected, page_html, test_report, test_failure
            )
        else:
            log.info(f"Attempting direct API hint with provider: {provider_name}")
            ai_response = _get_llm_root_cause_hint_direct_api(
                test_report, test_failure, context_collected, page_html, provider_name, provider_config
            )
        return ai_response.strip() if ai_response and not ai_response.startswith("Error:") else ai_response
    except Exception as e:
        log.exception("Unexpected error while requesting root cause hint (main function): %s", e)
        return f"Unexpected error during AI analysis: {e}"


def initialize_rag_system(provider_name: str, provider_config: dict, api_key_for_rag_if_needed: str = None):
    """Initialize the RAG system with LangChain for the specified provider."""
    if not provider_config.get("embedding_model"):
        raise ValueError(f"Missing 'embedding_model' in {provider_name} for RAG.")

    prompt_template_str = (
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
    prompt = ChatPromptTemplate.from_template(prompt_template_str)

    model_name = provider_config.get("model")
    embedding_model_name = provider_config.get("embedding_model")
    persist_directory = provider_config.get("persist_directory", "chroma_db")

    if provider_name == "ollama":
        embeddings = OllamaEmbeddings(model=embedding_model_name, base_url=provider_config.get("base_url"))
        llm = ChatOllama(
            model=model_name,
            temperature=provider_config.get("temperature", DEFAULT_TEMPERATURE),
            top_k=provider_config.get("top_k"),
            seed=provider_config.get("seed"),
            num_ctx=provider_config.get("num_ctx", DEFAULT_NUM_CONTEXT),
            base_url=provider_config.get("base_url")
        )
    elif provider_name == "gemini":
        # genai.configure should have been called by get_root_cause_hint if api_key_for_rag_if_needed was sourced
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=api_key_for_rag_if_needed)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key_for_rag_if_needed,
            temperature=provider_config.get("temperature", DEFAULT_TEMPERATURE),
            generation_config={"temperature": provider_config.get("temperature", DEFAULT_TEMPERATURE),
                               "top_k": provider_config.get("top_k")}
        )
    elif provider_name == "openai":
        embeddings = OpenAIEmbeddings(model=embedding_model_name, api_key=api_key_for_rag_if_needed)
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key_for_rag_if_needed,
            temperature=provider_config.get("temperature", DEFAULT_TEMPERATURE),
            model_kwargs={"top_p": provider_config.get("top_p")},
            max_tokens=provider_config.get("max_tokens_rag")
        )
    else:
        raise ValueError(f"Unsupported RAG provider: {provider_name}")

    try:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS_TO_RETRIEVE})
    except Exception as e:
        log.exception(f"Failed to initialize ChromaDB with persist_directory='{persist_directory}': {e}")
        raise

    def format_docs(docs):
        doc_result = "\n\n".join(doc.page_content for doc in docs)
        log.debug("Formatted documents for LLM: %s", doc_result)
        return doc_result

    rag_chain = (
            {"documentation": retriever | format_docs, "additional_context": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


def get_rag_response(rag_chain, context_collected:str, page_html:str, test_report:str, test_failure:str) -> str:
    """Get a RAG system response."""
    try:
        context = (
            f"- Test failure:\n{test_failure}\n\n"
            f"- Test report:\n{test_report}\n\n"
            f"- System logs:\n{context_collected}\n"
            f"- Pre-processed current HTML page:\n{page_html}"
        )
        ai_response = rag_chain.invoke(context)
        return ai_response.strip() if ai_response else "No response content received."
    except Exception as e:
        log.exception("Unexpected error during RAG chain invocation: %s", e)
        return f"Unexpected error during AI analysis (RAG): {e}"
