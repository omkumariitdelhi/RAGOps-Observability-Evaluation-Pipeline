import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import DATA_DIR, DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(data_dir: Path):
    documents = []
    # Load PDFs
    for pdf_path in data_dir.glob("*.pdf"):
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    
    # Load Markdown
    for md_path in data_dir.glob("*.md"):
        logger.info(f"Loading Markdown: {md_path}")
        loader = TextLoader(str(md_path), encoding="utf-8")
        documents.extend(loader.load())
        
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def ingest_to_chroma(chunks):
    logger.info(f"Initializing embeddings with {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    logger.info(f"Storing chunks into Chroma DB at {DB_DIR}...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )
    logger.info("Ingestion complete.")
    return db

def run_ingestion():
    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist.")
        return
    docs = load_documents(DATA_DIR)
    if not docs:
        logger.warning("No documents found to ingest.")
        return
    chunks = split_documents(docs)
    ingest_to_chroma(chunks)

if __name__ == "__main__":
    run_ingestion()
