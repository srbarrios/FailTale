import json
import logging
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import requests

# --- Configuration ---
PERSIST_DIRECTORY = "chroma_db"  # Folder where the vector database is stored
EMBEDDING_MODEL = "nomic-embed-text"  # Same embedding model used in indexing
LLM_MODEL = "granite3-dense:8b"  # LLM model in Ollama to generate responses
NUM_CHUNKS_TO_RETRIEVE = 50  # How many relevant chunks to search in the DB
# --- End Configuration ---

PROMPT_TEMPLATE = """
You are an expert QA analyst.
You are given the user guide documentation of a product:
--- BEGIN DATA ---
{documentation}
--- END DATA ---

You will receive a Gherkin-formatted test report with a failed scenario and its error stack trace, after this, you will receive the logs of the system:
{additional_context}

Use ALL the previous data to analyze the test failure and identify the most likely root cause of the failure.
Do not focus on xpath or selenium stack trace, instead focus on the gherkin steps and the logs of the system.
Be concise and factual.
"""

def initialize_rag_system():
    """Initializes the RAG system with LangChain."""

    # Load the embedding model
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Load the existing ChromaDB vector database
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    # Create the "Retriever"
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS_TO_RETRIEVE})

    # Load the LLM model from Ollama
    llm = ChatOllama(model=LLM_MODEL)

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Define the RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"documentation": retriever | format_docs, "additional_context": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain

def get_rag_response_with_ollama(rag_chain, context_collected, test_report, ollama_config):
    """Gets a response from the RAG system and sends it to Ollama for analysis."""
    if not ollama_config or not ollama_config.get('base_url'):
        logging.error("Ollama configuration missing or incomplete.")
        return "Error: Ollama configuration is not available."

    try:
        # Step 2: Configure the request to Ollama
        ollama_url = f"{ollama_config['base_url'].rstrip('/')}/api/chat"
        model = ollama_config.get('model', 'granite3-dense:8b')
        timeout = ollama_config.get('request_timeout', 60)

        context = f"""
        Test report:
        --- BEGIN DATA ---
        {test_report}
        --- END DATA ---
        Logs:
        --- BEGIN DATA ---
        {context_collected}
        --- END DATA ---
        """

        ai_response = rag_chain.invoke(context)
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

def get_root_cause_hint(context_collected, test_report, ollama_config):
    """Sends collected debug data to Ollama and requests a root cause hint."""

    try:
        rag_chain = initialize_rag_system()
        ai_response = get_rag_response_with_ollama(rag_chain, context_collected, test_report, ollama_config)

        return ai_response.strip() if ai_response else "No response content received."

    except Exception as e:
        logging.exception(f"Unexpected error while processing Ollama response: {e}")
        return f"Unexpected error during AI analysis: {e}"
