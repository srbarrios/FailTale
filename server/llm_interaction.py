# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import base64
import json
import logging
import re
import os
import sys
from typing import Optional, List, Dict, Any
import requests
from google import genai
from google.genai import types
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
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_K = 64
DEFAULT_TOP_P = 0.7
DEFAULT_MAX_TOKENS = -2 # -2 means no limit for OpenAI, Ollama, and Gemini
DEFAULT_MAX_OUTPUT_TOKENS = 1024 # For Gemini explicit output limit
DEFAULT_SEED = 42
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_NUM_CONTEXT = 2048 # Default for Ollama if not in config

# --- Helper to validate LLM config ---
def _validate_llm_provider_config(provider_config: dict, provider_name: str) -> bool:
    if not provider_config:
        log.error(f"{provider_name} configuration settings missing.")
        return False
    if not provider_config.get("model"):
        log.error(f"LLM provider '{provider_name}' missing 'model' in its settings.")
        return False
    if provider_name == "ollama" and not provider_config.get("base_url"):
        log.error("Ollama settings missing 'base_url'.")
        return False
    if provider_name in {"gemini", "openai"}:
        api_key = provider_config.get("api_key")
        api_key_env_var = provider_config.get("api_key_env_var")
        if not api_key:
            default_env = "GOOGLE_API_KEY" if provider_name == "gemini" else "OPENAI_API_KEY"
            env_var_to_check = api_key_env_var or default_env
            if not os.getenv(env_var_to_check):
                log.error(
                    f"{provider_name.capitalize()} API key not found in config ('api_key') "
                    f"and environment variable '{env_var_to_check}' not set."
                )
                return False
    return True

# --- Unified LLM Interaction Function ---
def _call_llm_api(
        text_messages: List[Dict[str, str]],
        provider_name: str,
        provider_config: dict,
        screenshot_b64_str: Optional[str] = None,
        image_mime_type: str = "image/png"
) -> str:
    if not _validate_llm_provider_config(provider_config, provider_name):
        return f"Error: Invalid {provider_name} configuration."

    model_name = provider_config.get("model")
    log.info(f"Calling LLM provider: {provider_name}, Model: {model_name}")

    if provider_name == "ollama":
        ollama_url = f"{provider_config['base_url'].rstrip('/')}/api/chat"
        timeout = int(provider_config.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))
        ollama_messages = [msg.copy() for msg in text_messages]
        is_ollama_multimodal = provider_config.get("multimodal", False)

        if screenshot_b64_str and is_ollama_multimodal:
            if ollama_messages and ollama_messages[-1]["role"] == "user":
                ollama_messages[-1]["images"] = [screenshot_b64_str]
            else: # Create a new message if no user message or last one is not user
                ollama_messages.append({"role": "user", "content": "Extract and analyze the visible text. Correlate the extracted text with the test failure and test report.", "images": [screenshot_b64_str]})
            log.debug("Screenshot added for Ollama multimodal model.")

        options = {
            "temperature": provider_config.get("temperature", DEFAULT_TEMPERATURE),
            "num_ctx": provider_config.get("num_ctx", DEFAULT_NUM_CONTEXT),
            "seed": provider_config.get("seed", DEFAULT_SEED),
            "top_k": provider_config.get("top_k", DEFAULT_TOP_K),
            "max_tokens": provider_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        }
        options = {k: v for k, v in options.items() if v is not None}
        payload = {"model": model_name, "messages": ollama_messages, "options": options, "stream": False}
        log.debug(f"Ollama API Request: {json.dumps(payload, indent=2, default=lambda x: '<image_data_omitted>' if x == screenshot_b64_str else str(x))}")
        try:
            response = requests.post(ollama_url, json=payload, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("message", {}).get("content", "")
        except Exception as e:
            log.exception(f"Error during Ollama API call to {ollama_url}: {e}")
            return f"Error: Ollama API call failed - {e}"

    elif provider_name == "gemini":
        api_key = provider_config.get("api_key") or os.getenv(provider_config.get("api_key_env_var", "GOOGLE_API_KEY"))
        if not api_key: return "Error: Gemini API key missing."
        try:
            client = genai.Client(api_key=api_key)
            gemini_content_parts = []
            full_text_prompt = ""
            if text_messages:
                for msg in text_messages:
                    if msg.get("role") == "system":
                        full_text_prompt += f"System Instructions: {msg['content']}\n\n"
                    else: # user, assistant
                        full_text_prompt += f"{msg['content']}\n\n"
            if full_text_prompt:
                gemini_content_parts.append(full_text_prompt.strip())

            is_gemini_multimodal = provider_config.get("multimodal", False)
            if screenshot_b64_str and is_gemini_multimodal:
                try:
                    gemini_content_parts.append("Extract and analyze the visible text of the following image, and correlate the extracted text with the test failure and test report.")
                    gemini_content_parts.append(genai.types.Part.from_bytes(data=base64.b64decode(screenshot_b64_str), mime_type=image_mime_type))
                    log.debug("Added screenshot as image part for Gemini.")
                except Exception as e:
                    log.error(f"Error preparing image for Gemini: {e}. Appending as text.")

            if not gemini_content_parts:
                log.warning("No content parts to send to Gemini.")
                return "Error: No content for Gemini."

            generation_config_params = {
                "temperature": provider_config.get("temperature", DEFAULT_TEMPERATURE),
                "top_k": provider_config.get("top_k", DEFAULT_TOP_K),
                "max_output_tokens": provider_config.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
            }
            generation_config = genai.types.GenerateContentConfig(
                **{k:v for k,v in generation_config_params.items() if v is not None}
            )
            log.debug(f"Gemini API Request: Model='{model_name}', Config={generation_config_params}, Parts Count={len(gemini_content_parts)}")
            response = client.models.generate_content(model=model_name, contents=gemini_content_parts, config=generation_config)

            if not response.candidates or not hasattr(response, 'text') or not response.text:
                feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'
                log.error(f"Gemini API: No candidates or text. Feedback: {feedback}")
                reason = feedback.block_reason if hasattr(feedback, 'block_reason') and feedback.block_reason else "Unknown"
                return f"Error: Gemini API - No content. Reason: {reason}"
            return response.text
        except Exception as e:
            log.exception(f"Unexpected error with Gemini API call: {e}")
            return f"Error: Unexpected error with Gemini: {e}"

    elif provider_name == "openai":
        api_key = provider_config.get("api_key") or os.getenv(provider_config.get("api_key_env_var", "OPENAI_API_KEY"))
        if not api_key: return "Error: OpenAI API key missing."

        client = OpenAI(api_key=api_key)
        timeout = float(provider_config.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))

        try:
            openai_final_messages = []
            if text_messages:
                for msg in text_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role not in ["system", "user", "assistant"]: role = "user"
                    openai_final_messages.append({"role": role, "content": content})

            is_openai_multimodal = provider_config.get("multimodal", False)
            if screenshot_b64_str and is_openai_multimodal:
                image_content_part = {"type": "image_url", "image_url": {"url": f"data:{image_mime_type};base64,{screenshot_b64_str}"}}
                # Append image to the content of the last user message, or create a new one
                if openai_final_messages and openai_final_messages[-1]["role"] == "user":
                    last_user_content = openai_final_messages[-1]["content"]
                    if isinstance(last_user_content, str):
                        openai_final_messages[-1]["content"] = [{"type": "text", "text": last_user_content}, image_content_part]
                    elif isinstance(last_user_content, list):
                        openai_final_messages[-1]["content"].append(image_content_part)
                else: # No user message to append to, or the last message isn't user; create a new one
                    openai_final_messages.append({"role": "user", "content": [
                        {"type": "text", "text": "Extract and analyze the visible text. Correlate the extracted text with the test failure and test report."},
                        image_content_part
                    ]})
                log.debug("Added screenshot to OpenAI messages.")

            if not openai_final_messages:
                log.warning("No messages to send to OpenAI.")
                return "Error: No messages for OpenAI."

            completion_params = {
                "model": model_name,
                "messages": openai_final_messages,
                "temperature": provider_config.get("temperature", DEFAULT_TEMPERATURE),
                "top_p": provider_config.get("top_p", DEFAULT_TOP_P),
                "max_tokens": provider_config.get("max_tokens") if provider_config.get("max_tokens") != DEFAULT_MAX_TOKENS else None
            }
            completion_params = {k: v for k, v in completion_params.items() if v is not None}

            log.debug(f"OpenAI API Request: Model='{model_name}', Params={completion_params}")
            response = client.chat.completions.create(**completion_params, timeout=timeout)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                finish_reason = response.choices[0].finish_reason if response.choices and response.choices[0] else 'N/A'
                log.error(f"OpenAI API: No choices or message content. Finish reason: {finish_reason}")
                return "Error: OpenAI API - No content in response."
        except Exception as e:
            log.exception(f"Unexpected error with OpenAI API call: {e}")
            return f"Error: Unexpected error with OpenAI: {e}"
    else:
        log.error(f"Unsupported LLM provider: {provider_name}")
        return "Error: Unsupported LLM provider."

def get_hosts_to_collect(hosts: list, test_report: str, provider_name: str, provider_config: dict) -> list:
    if not _validate_llm_provider_config(provider_config, provider_name):
        log.error(f"Invalid configuration for provider '{provider_name}' in get_hosts_to_collect.")
        return []

    prompt_content = (
        f"You are a test automation assistant.\n"
        f"Available hosts: {json.dumps(hosts)}\n\n"
        f"Test report: {test_report}\n\n"
        f"Return a comma-separated list of **up to 3 hostnames** from the available hosts.\n"
        f"Choose those that most likely match words from the test report.\n"
        f"Respond ONLY with hostnames separated by commas. No explanation or extra formatting."
    )
    text_messages = [{"role": "user", "content": prompt_content}]
    full_message = _call_llm_api(text_messages, provider_name, provider_config)

    if full_message is None or full_message.startswith("Error:"):
        log.error(f"Failed to get host list from LLM ({provider_name}): {full_message}")
        return []

    log.debug(f"LLM Response for host selection ({provider_name}): {full_message}")
    hostnames_found = re.findall(r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}\b', full_message)
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

# Parameter screenshot_b64_str is now expected
def _get_llm_root_cause_hint_direct_api(test_report:str, test_failure:str, context_collected:str, screenshot_b64_str: Optional[str], provider_name: str, provider_config: dict) -> str:
    if not _validate_llm_provider_config(provider_config, provider_name):
        log.error(f"Invalid configuration for provider '{provider_name}' in _get_llm_root_cause_hint_direct_api.")
        return f"Error: Invalid {provider_name} configuration."

    prompt_text_parts = [
        provider_config.get("base_prompt", ""),
        "You are a QA Analyst. You will analyze data from a failed automated test.",
        "Your task is to provide a hint that identifies the most likely root cause.\n",
        "You are given:\n",
        f"- The test failure message: {test_failure}\n\n",
        f"- The full Gherkin test report: {test_report}\n\n",
        f"- System logs from the test environment: {context_collected}\n\n",
        # Screenshot will be handled by _call_llm_api if screenshot_b64_str is provided
    ]
    prompt_text_parts.extend([
        "Instructions:\n",
        "- Look for exact word matches between the failure message and other data.\n",
        "- Prioritize logs with the keywords 'error', 'err', or error codes.\n",
        "- Don't output: introduction, summary or instructions.\n",
        "- Give facts (example: Description and Output logs related to the hint)\n",
        "Hint:"
    ])
    final_prompt_content = "".join(prompt_text_parts)
    text_messages = [{"role": "user", "content": final_prompt_content}]

    return _call_llm_api(text_messages, provider_name, provider_config, screenshot_b64_str=screenshot_b64_str)

# Parameter screenshot_b64_str is now expected
def get_root_cause_hint(context_collected:str, test_report:str, test_failure:str, screenshot_b64_str: Optional[str], provider_name: str, provider_config: dict, with_rag:bool=False) -> str:
    if not _validate_llm_provider_config(provider_config, provider_name):
        log.error(f"Invalid configuration for provider '{provider_name}' in get_root_cause_hint.")
        return f"Error: Invalid {provider_name} configuration."

    try:
        if with_rag:
            log.info(f"Attempting RAG-based hint with provider: {provider_name}")
            api_key_for_rag = provider_config.get("api_key")
            if not api_key_for_rag and provider_name in ["gemini", "openai"]:
                default_env_var = "GOOGLE_API_KEY" if provider_name == "gemini" else "OPENAI_API_KEY"
                api_key_env_var_name = provider_config.get("api_key_env_var", default_env_var)
                api_key_for_rag = os.getenv(api_key_env_var_name)
            if provider_name in ["gemini", "openai"] and not api_key_for_rag:
                log.error(f"{provider_name.capitalize()} API key not found for RAG.")
                return f"Error: {provider_name.capitalize()} API key not set for RAG."
            if provider_name == "gemini" and api_key_for_rag:
                genai.configure(api_key=api_key_for_rag)

            rag_chain = initialize_rag_system(provider_name, provider_config, api_key_for_rag_if_needed=api_key_for_rag)
            ai_response = get_rag_response(
                rag_chain, context_collected, test_report, test_failure
            )
        else:
            log.info(f"Attempting direct API hint with provider: {provider_name}")
            ai_response = _get_llm_root_cause_hint_direct_api(
                test_report, test_failure, context_collected, screenshot_b64_str, provider_name, provider_config
            )
        return ai_response.strip() if ai_response and not ai_response.startswith("Error:") else ai_response
    except Exception as e:
        log.exception("Unexpected error while requesting root cause hint (main function): %s", e)
        return f"Unexpected error during AI analysis: {e}"

def initialize_rag_system(provider_name: str, provider_config: dict, api_key_for_rag_if_needed: Optional[str] = None):
    if not provider_config.get("embedding_model"):
        raise ValueError(f"Missing 'embedding_model' in {provider_name} for RAG.")

    prompt_template_str = (
        "You are a QA Analyst. You will analyze data from a failed automated test. "
        "Your task is to provide a hint that identifies the most likely root cause.\n"
        "You are given:\n"
        "1. Test details (including logs, failure messages):\n" # Screenshot related text removed
        "|{additional_context}|\n\n"
        "2. Relevant documentation:|{documentation}|\n\n"
        "Instructions:\n"
        "- Focus on exact word matches between the test failure and logs.\n"
        "- Focus on exact word matches between the test report and the documentation.\n"
        "- Don't output: introduction, summary or instructions.\n\n"
        "- Give facts (example: Description and Output logs related to the hint)\n"
        "Hint:\n"
    )
    prompt = ChatPromptTemplate.from_template(provider_config.get("base_prompt", "").join(prompt_template_str))
    model_name = provider_config.get("model")
    embedding_model_name = provider_config.get("embedding_model")
    persist_directory = provider_config.get("persist_directory", "chroma_db")
    llm, embeddings = None, None

    if provider_name == "ollama":
        base_url = provider_config.get("base_url")
        embeddings = OllamaEmbeddings(model=embedding_model_name, base_url=base_url)
        llm = ChatOllama(
            model=model_name, base_url=base_url,
            temperature=provider_config.get("temperature", DEFAULT_TEMPERATURE),
            top_k=provider_config.get("top_k", DEFAULT_TOP_K),
            seed=provider_config.get("seed", DEFAULT_SEED),
            num_ctx=provider_config.get("num_ctx", DEFAULT_NUM_CONTEXT),
        )
    elif provider_name == "gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=api_key_for_rag_if_needed)
        llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key_for_rag_if_needed,
            temperature=provider_config.get("temperature", DEFAULT_TEMPERATURE),
            generation_config={"temperature": provider_config.get("temperature", DEFAULT_TEMPERATURE),
                               "top_k": provider_config.get("top_k", DEFAULT_TOP_K)}
        )
    elif provider_name == "openai":
        embeddings = OpenAIEmbeddings(model=embedding_model_name, api_key=api_key_for_rag_if_needed)
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key_for_rag_if_needed,
            temperature=provider_config.get("temperature", DEFAULT_TEMPERATURE),
            model_kwargs={"top_p": provider_config.get("top_p", DEFAULT_TOP_P)},
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
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
            {"documentation": retriever | format_docs, "additional_context": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain

# Parameter screenshot_b64_str is now expected
def get_rag_response(rag_chain, context_collected:str, test_report:str, test_failure:str) -> str:
    """Get a RAG system response. Screenshot is NOT directly processed as an image by this RAG setup."""
    try:
        context_parts = [
            f"- Test failure:\n{test_failure}\n\n",
            f"- Test report:\n{test_report}\n\n",
            f"- System logs:\n{context_collected}\n",
        ]

        context = "".join(context_parts)
        ai_response = rag_chain.invoke(context)
        return ai_response.strip() if ai_response else "No response content received."
    except Exception as e:
        log.exception("Unexpected error during RAG chain invocation: %s", e)
        return f"Unexpected error during AI analysis (RAG): {e}"
