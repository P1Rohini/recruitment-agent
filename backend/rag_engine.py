"""
RAG Engine — handles document ingestion, chunking, embedding, and retrieval.
Uses LOCAL sentence-transformers embeddings (no API key needed, runs on your machine).
LLM calls still go to Gemini — only embeddings are local.
"""

import os
import fitz  # PyMuPDF
import docx
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_PATH = "./vectorstore"

print("[RAG] Loading local embedding model (all-MiniLM-L6-v2)...")
print("[RAG] First run will download ~90MB — please wait...")

from sentence_transformers import SentenceTransformer
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

print("[RAG] Embedding model ready ✓")


class LocalEmbeddings(Embeddings):
    """
    Local CPU-based embeddings using sentence-transformers.
    Model: all-MiniLM-L6-v2
    - Free, no API key, no internet after first download
    - 90MB model, 384-dimensional vectors
    - Fast on CPU, good quality for English text
    - Industry standard for RAG on private infrastructure
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = _EMBED_MODEL.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = _EMBED_MODEL.encode([text], show_progress_bar=False)
        return vector[0].tolist()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract raw text from a Word document."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def extract_text(file_path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_document(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """
    Split text into overlapping chunks.
    RecursiveCharacterTextSplitter tries paragraph -> sentence -> word boundaries.
    chunk_size=600: enough context per chunk without losing precision.
    chunk_overlap=100: prevents losing meaning at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents([text], metadatas=[metadata])
    return chunks


def get_vectorstore(collection_name: str = "resumes") -> Chroma:
    """Get or create a ChromaDB vector store with local embeddings."""
    embeddings = LocalEmbeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_PATH,
    )
    return vectorstore


def ingest_document(file_path: str, doc_type: str, doc_id: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
    1. Extract text from file
    2. Chunk with overlap
    3. Embed locally using sentence-transformers
    4. Store in ChromaDB
    """
    text = extract_text(file_path)
    if not text.strip():
        raise ValueError("Document appears to be empty or unreadable.")

    metadata = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "file_name": Path(file_path).name,
        "file_path": file_path,
    }

    chunks = chunk_document(text, metadata)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    return {
        "doc_id": doc_id,
        "file_name": Path(file_path).name,
        "chunk_count": len(chunks),
        "char_count": len(text),
    }


def retrieve_chunks(query: str, doc_id: str = None, k: int = 6) -> List[Document]:
    """
    Semantic retrieval: find top-K chunks relevant to query.
    Optionally filter by doc_id to search within a specific document.
    """
    vectorstore = get_vectorstore()
    search_kwargs = {"k": k}
    if doc_id:
        search_kwargs["filter"] = {"doc_id": doc_id}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


def list_documents() -> List[Dict[str, Any]]:
    """List all ingested documents from ChromaDB metadata."""
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    results = collection.get(include=["metadatas"])

    seen = {}
    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "doc_type": meta.get("doc_type"),
                "file_name": meta.get("file_name"),
            }
    return list(seen.values())


def delete_document(doc_id: str) -> bool:
    """Remove all chunks of a document from the vector store."""
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    results = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        return True
    return False