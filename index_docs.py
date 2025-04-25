# Copyright (c) 2025 Oscar Barrios
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import shutil
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from langchain_chroma import Chroma
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_ollama import OllamaEmbeddings

# Filter the correct warning shown in the traceback
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# --- Configuration ---
SOURCE_DIRECTORY = "/Users/obarrios/Desktop/docs"  # Folder with your HTML docs
PERSIST_DIRECTORY = "examples/uyuni/chroma_db"      # Where ChromaDB will be saved
EMBEDDING_MODEL = "nomic-embed-text"                # Ollama embedding model
CHUNK_SIZE = 1800                                   # Max size of text chunks (in characters)
CHUNK_OVERLAP = 300                                 # Overlap between chunks (in characters)
# --- End Configuration ---

def split_html_by_headings(html_text, source_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits HTML text into chunks based on h1, h2, h3 headings.
    Handles cases where no headings are found by treating the whole content as one section.
    """
    soup = BeautifulSoup(html_text, "lxml")
    chunks = []
    headings = soup.find_all(['h1', 'h2', 'h3'])

    if not headings:
        # If no headings, treat the whole document body text as one chunk (or split if too large)
        body_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
        if not body_text: # Fallback if no body or body is empty
            body_text = soup.get_text(separator=" ", strip=True)

        if body_text:
            section_title = f"Full Document: {os.path.basename(source_path)}" # Use filename as fallback title
            if len(body_text) <= chunk_size:
                chunks.append((section_title, body_text))
            else:
                # Content too large, split it into overlapping chunks
                start = 0
                while start < len(body_text):
                    end = start + chunk_size
                    chunk_text = body_text[start:end]
                    chunks.append((section_title, chunk_text))
                    start += chunk_size - chunk_overlap
        return chunks # Return early if no headings found


    # Process sections based on headings
    current_heading_text = f"Introduction: {os.path.basename(source_path)}" # Content before the first heading
    current_content_pieces = []

    # Handle content before the first heading
    for elem in headings[0].find_previous_siblings():
        current_content_pieces.insert(0, elem.get_text(separator=" ", strip=True)) # Insert at beginning to maintain order

    # Process each heading and the content following it
    for i, heading in enumerate(headings):
        # Finalize the previous section
        section_content = current_heading_text + "\n\n" + "\n".join(current_content_pieces).strip()
        if section_content.strip(): # Add chunk only if there's content
            if len(section_content) <= chunk_size:
                chunks.append((current_heading_text.split('\n\n')[0], section_content)) # Use only heading text for title
            else:
                start = 0
                while start < len(section_content):
                    end = start + chunk_size
                    chunk_text = section_content[start:end]
                    chunks.append((current_heading_text.split('\n\n')[0], chunk_text))
                    start += chunk_size - chunk_overlap

        # Start the new section
        current_heading_text = heading.get_text(separator=" ", strip=True)
        current_content_pieces = [] # Reset content for the new section

        # Gather content between this heading and the next
        for sibling in heading.find_next_siblings():
            if sibling.name and sibling.name in ['h1', 'h2', 'h3']:
                break # Stop when the next heading is found
            current_content_pieces.append(sibling.get_text(separator=" ", strip=True))

    # Add the last section (after the last heading)
    section_content = current_heading_text + "\n\n" + "\n".join(current_content_pieces).strip()
    if section_content.strip(): # Add chunk only if there's content
        if len(section_content) <= chunk_size:
            chunks.append((current_heading_text.split('\n\n')[0], section_content))
        else:
            start = 0
            while start < len(section_content):
                end = start + chunk_size
                chunk_text = section_content[start:end]
                chunks.append((current_heading_text.split('\n\n')[0], chunk_text))
                start += chunk_size - chunk_overlap

    return chunks


def main():
    print("Starting indexing process...")

    # Delete old database if exists
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Deleting old database in '{PERSIST_DIRECTORY}'...")
        shutil.rmtree(PERSIST_DIRECTORY)

    # Load HTML documents - Using BSHTMLLoader to ensure full HTML content per doc
    print(f"Loading documents from '{SOURCE_DIRECTORY}'...")
    loader = DirectoryLoader(
        SOURCE_DIRECTORY,
        glob="**/*.html",
        # Use BSHTMLLoader to get full HTML content per document
        # It reads the file and passes the content to BeautifulSoup
        loader_cls=BSHTMLLoader,
        loader_kwargs={'open_encoding': 'utf-8'}, # Specify encoding if needed
        show_progress=True,
        use_multithreading=True,
        # Remove UnstructuredHTMLLoader specific kwargs if using BSHTMLLoader
        # loader_kwargs={'mode': 'elements', 'strategy': 'fast'}
    )

    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Please ensure the path exists and contains valid HTML files.")
        return

    if not documents:
        print(f"Error: No HTML documents found or loaded from '{SOURCE_DIRECTORY}'.")
        return
    print(f"{len(documents)} documents loaded.")

    # Split and prepare chunks
    print("Splitting documents by HTML headings...")
    split_documents = []

    for doc in documents:
        if not isinstance(doc.page_content, str) or not doc.page_content.strip():
            print(f"Warning: Skipping document with empty or non-string content: {doc.metadata.get('source', 'Unknown source')}")
            continue

        html_text = doc.page_content
        metadata = doc.metadata
        source_path = metadata.get('source', 'Unknown Source') # Get source path for better titles

        # Pass source_path to potentially use in titles if no headings found
        section_chunks = split_html_by_headings(html_text, source_path, CHUNK_SIZE, CHUNK_OVERLAP)

        if not section_chunks:
            print(f"Warning: No chunks generated for document: {source_path}")
            # Optionally, add the whole document if splitting failed but content exists
            # plain_text = BeautifulSoup(html_text, "lxml").get_text(separator=" ", strip=True)
            # if plain_text:
            #    split_documents.append(Document(page_content=plain_text, metadata=metadata))
            continue # Skip doc if no chunks and not adding whole doc

        for heading_title, chunk_text in section_chunks:
            if not chunk_text.strip(): # Skip empty chunks
                continue
            new_metadata = metadata.copy()
            new_metadata['section_title'] = heading_title
            split_documents.append(Document(page_content=chunk_text, metadata=new_metadata))

    print(f"Generated {len(split_documents)} chunks ready for embedding.")

    # --- Add check for empty chunks before embedding ---
    if not split_documents:
        print("Error: No text chunks were generated from the documents. Cannot proceed with embedding.")
        print("Please check the HTML structure of your documents in '{SOURCE_DIRECTORY}' and the 'split_html_by_headings' function.")
        return
    # --- End check ---

    # Create embeddings and store them
    print(f"Embedding chunks using '{EMBEDDING_MODEL}' model...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        vectorstore = Chroma.from_documents(
            documents=split_documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )

        print(f"Indexing completed successfully! Vector database saved and persisted at '{PERSIST_DIRECTORY}'.")

    except Exception as e:
        print(f"Error during embedding or storing in ChromaDB: {e}")
        # Add more specific error handling if needed based on potential Chroma/Ollama errors


if __name__ == "__main__":
    main()
