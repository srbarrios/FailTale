import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
SOURCE_DIRECTORY = "/Users/obarrios/Desktop/docs"  # Folder containing your HTML files
PERSIST_DIRECTORY = "chroma_db"  # Folder where the vector database will be saved
EMBEDDING_MODEL = "nomic-embed-text"  # Name of the embedding model in Ollama
CHUNK_SIZE = 3000  # Size of text chunks (in characters)
CHUNK_OVERLAP = 200  # Overlap between chunks (in characters)
# --- End Configuration ---

def main():
    print("Starting indexing process...")

    # Check if the persistence directory exists and delete it if necessary
    # This ensures we index from scratch every time the script is run.
    # You can comment out these lines if you want to add documents incrementally (more advanced).
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Deleting old database in '{PERSIST_DIRECTORY}'...")
        shutil.rmtree(PERSIST_DIRECTORY)

    # 1. Load HTML documents
    print(f"Loading documents from '{SOURCE_DIRECTORY}'...")
    # We use BSHTMLLoader to properly parse HTML
    loader = DirectoryLoader(
        SOURCE_DIRECTORY,
        glob="**/*.html",  # Loads all .html files recursively
        loader_cls=BSHTMLLoader,
        show_progress=True,
        use_multithreading=True,  # Speeds up loading if you have many files
        loader_kwargs={'open_encoding': 'utf-8', 'bs_kwargs': {'features': 'lxml'}}  # Ensures UTF-8 and uses lxml parser
    )
    documents = loader.load()
    if not documents:
        print("Error: No HTML documents found in the specified folder.")
        return
    print(f"{len(documents)} documents loaded.")

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,  # Helps identify the origin of each chunk
    )
    splits = text_splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} chunks.")

    # 3. Create embeddings and store them in ChromaDB
    print(f"Creating embeddings using the '{EMBEDDING_MODEL}' model and storing in ChromaDB...")
    # Initialize the embedding model from Ollama
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Create the ChromaDB vector database from the chunks
    # It will be saved in the folder specified in PERSIST_DIRECTORY
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print(f"Indexing completed! Vector database saved in '{PERSIST_DIRECTORY}'.")

if __name__ == "__main__":
    main()
