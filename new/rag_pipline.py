"""
rag_pipline.py
-----------------

This module defines the classes and functions related to ingesting legal PDF
documents and storing their semantic representations in a vector database.  It
encapsulates the Retrieval‑Augmented Generation (RAG) portion of the system.

The main class exported by this module is ``DocumentIngestor``.  Given a
directory of PDF files and a unique case identifier, it will extract
paragraph‑level text snippets from each document, compute embeddings using
Azure OpenAI, and upsert them into a Chroma vector database collection.

The ``DocumentChunk`` model provides a simple container for the text and
associated metadata returned from semantic searches.

Note:  You must configure your Azure OpenAI credentials via the environment
variables ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT`` before
instantiating the ``DocumentIngestor``.  See the README for details.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import os
import subprocess
import numpy as np
from pydantic import BaseModel

try:
    # Optional: use chromadb if available.  This library provides a
    # persistent vector store and can speed up similarity search.  If it
    # isn't installed, we fall back to a simple in‑memory implementation
    # implemented in this module.
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    from chromadb.utils import embedding_functions  # type: ignore
    _HAVE_CHROMA = True
except ImportError:
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    embedding_functions = None  # type: ignore
    _HAVE_CHROMA = False

try:
    import openai  # type: ignore
except ImportError:
    # ``openai`` is required to compute embeddings when chromadb isn't
    # available.  It must be installed by the user.  We don't import it
    # immediately to allow the rest of the module to be imported without
    # errors in environments lacking openai.
    openai = None  # type: ignore


class DocumentChunk(BaseModel):
    """Represents a chunk of text extracted from a document along with metadata."""

    id: str
    case_id: str
    doc_name: str
    page_number: int
    text: str
    metadata: Dict[str, Any]


class DocumentIngestor:
    """
    Handles PDF ingestion, chunking, embedding generation, and storage.  By
    default this class stores all embeddings in‑memory and uses the
    OpenAI API for generating embeddings.  If the optional ``chromadb``
    package is installed, it will instead create a persistent vector
    collection and delegate embedding management to Chroma.  When using
    Chroma the Azure OpenAI embedding function from ``chromadb`` is no
    longer used—instead embeddings are computed via OpenAI's standard
    API (see below).

    Set the ``OPENAI_API_KEY`` environment variable with your OpenAI API
    key before instantiating this class.  You can also specify the
    embedding model via the ``OPENAI_EMBEDDING_MODEL`` environment
    variable; the default is ``text-embedding-ada-002``.
    """

    def __init__(self, collection_name: str = "legal_docs", embedding_model: str | None = None) -> None:
        # Determine whether to use Chroma based on availability
        self.use_chroma = _HAVE_CHROMA
        self.collection_name = collection_name
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

        # In‑memory storage for document chunks and their embeddings
        self._chunks: List[Tuple[DocumentChunk, List[float]]] = []

        # Configure OpenAI API key.  Note: we only set ``openai.api_key`` if the
        # ``openai`` package was imported successfully.  The user must
        # ensure that the package is installed and the API key is provided.
        if openai:
            openai.api_key = os.getenv("OPENAI_API_KEY", "")

        if self.use_chroma:
            # Initialise Chroma client and collection.  We still manage our own
            # embeddings but store them in Chroma for efficient search.  If
            # a collection with the same name exists it will be reset.
            self.client = chromadb.Client(Settings(allow_reset=True))  # type: ignore
            if collection_name in [c.name for c in self.client.list_collections()]:  # type: ignore
                self.client.delete_collection(collection_name)  # type: ignore
            self.collection = self.client.create_collection(collection_name)  # type: ignore

    def _embed_text(self, text: str) -> List[float]:
        """Compute an embedding for a single piece of text via OpenAI."""
        if not openai:
            raise RuntimeError(
                "The openai package is required for embedding generation. Please install it and set OPENAI_API_KEY."
            )
        response = openai.Embedding.create(input=[text], model=self.embedding_model)  # type: ignore
        return response["data"][0]["embedding"]  # type: ignore

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file using the ``pdftotext`` command‑line
        utility.  This function reads the entire content of the PDF and
        returns it as a single string.  If pdftotext is not available the
        user should install poppler on their system.
        """
        result = subprocess.run(
            ["pdftotext", "-layout", file_path, "-"], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftotext failed for {file_path}: {result.stderr}")
        return result.stdout

    def ingest_case_documents(self, case_id: str, docs_dir: str) -> None:
        """
        Ingest all PDF documents for a given case.  This method will iterate
        through every PDF file in ``docs_dir``, extract the text using
        ``pdftotext``, split it into paragraph‑level chunks, compute
        embeddings for each paragraph and either store them in memory or
        upsert them into a Chroma collection.

        Parameters
        ----------
        case_id: str
            A unique identifier for the case whose documents are being
            ingested.
        docs_dir: str
            Path to a directory containing PDF files.  Non‑PDF files are
            ignored.
        """
        if not os.path.isdir(docs_dir):
            raise FileNotFoundError(f"Document directory not found: {docs_dir}")

        for filename in os.listdir(docs_dir):
            if not filename.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(docs_dir, filename)
            # Extract all text from the PDF
            full_text = self._extract_text_from_pdf(file_path)
            # Split by double newlines to approximate paragraphs
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            for idx, paragraph in enumerate(paragraphs):
                chunk_id = f"{case_id}:{filename}:{idx}"
                metadata: Dict[str, Any] = {
                    "case_id": case_id,
                    "document": filename,
                    "paragraph_index": idx,
                }
                # Compute embedding
                embedding = self._embed_text(paragraph)
                chunk = DocumentChunk(
                    id=chunk_id,
                    case_id=case_id,
                    doc_name=filename,
                    page_number=0,  # page number unavailable via pdftotext
                    text=paragraph,
                    metadata=metadata,
                )
                if self.use_chroma:
                    # Upsert into Chroma collection.  Note: Chroma expects lists
                    self.collection.add(  # type: ignore
                        documents=[paragraph],
                        metadatas=[metadata],
                        ids=[chunk_id],
                        embeddings=[embedding],
                    )
                else:
                    # Store in memory
                    self._chunks.append((chunk, embedding))

    def query(self, query_text: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Perform a semantic search over the ingested documents and return
        ``top_k`` most relevant chunks along with their metadata.  When
        using Chroma the query is delegated to the underlying Chroma
        collection.  Otherwise an in‑memory similarity search is
        performed using cosine similarity.
        """
        if self.use_chroma:
            results = self.collection.query(  # type: ignore
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks: List[DocumentChunk] = []
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                chunks.append(
                    DocumentChunk(
                        id=f"{metadata.get('case_id')}:{metadata.get('document')}",
                        case_id=metadata.get("case_id"),
                        doc_name=metadata.get("document"),
                        page_number=metadata.get("page", 0),
                        text=doc,
                        metadata=metadata,
                    )
                )
            return chunks

        # In‑memory search: compute embedding for the query
        query_vec = np.array(self._embed_text(query_text))
        # Compute cosine similarity with each stored embedding
        scored: List[Tuple[float, DocumentChunk]] = []
        for chunk, emb in self._chunks:
            vec = np.array(emb)
            # Avoid division by zero
            denom = np.linalg.norm(vec) * np.linalg.norm(query_vec)
            similarity = float(np.dot(vec, query_vec) / denom) if denom != 0 else 0.0
            scored.append((similarity, chunk))
        # Sort by similarity descending and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]
