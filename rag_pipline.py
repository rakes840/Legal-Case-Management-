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
from typing import Any, Dict, List

import fitz  # PyMuPDF for PDF processing
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pydantic import BaseModel


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
    Handles PDF ingestion, chunking, embedding generation, and storage in a
    vector database (Chroma).

    Each instance of ``DocumentIngestor`` manages its own Chroma client and
    collection.  When instantiating the class you may specify the name of
    the collection; by default ``legal_docs`` is used.  If a collection of
    that name already exists it will be reset, ensuring idempotent runs.

    The embedding function is configured to use Azure OpenAI.  Make sure
    environment variables are set correctly for ``AZURE_OPENAI_API_KEY`` and
    ``AZURE_OPENAI_ENDPOINT`` or pass them directly when creating the
    ``DocumentIngestor``.  You can override the deployment name via the
    ``embedding_deployment`` parameter.
    """

    def __init__(
        self,
        collection_name: str = "legal_docs",
        embedding_deployment: str | None = None,
    ) -> None:
        # Create a Chroma client and reset any existing collection with the
        # given name.  ``allow_reset`` enables deletion of existing
        # collections.
        self.client = chromadb.Client(Settings(allow_reset=True))
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(collection_name)
        self.collection = self.client.create_collection(collection_name)

        # Configure the embedding function using Azure OpenAI.  If a custom
        # deployment name is not provided the default for the environment will
        # be used.
        deployment_name = embedding_deployment or os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
        )
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base=api_base,
            api_type="azure",
            api_version="2023-05-15",
            deployment_name=deployment_name,
        )

    def ingest_case_documents(self, case_id: str, docs_dir: str) -> None:
        """
        Ingest all PDF documents for a given case.

        This method will iterate through every PDF file in ``docs_dir``, extract
        the text on a page‑by‑page basis, further split the text into
        paragraph‑level chunks, compute embeddings for each paragraph and
        upsert the resulting vectors into the Chroma collection.  The ``id``
        assigned to each chunk encodes the case ID, filename, page number and
        paragraph index.

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
            with fitz.open(file_path) as doc:
                for page_number in range(doc.page_count):
                    page = doc[page_number]
                    # Extract text from the page
                    raw_text = page.get_text().strip()
                    if not raw_text:
                        continue
                    # Chunk the page by paragraphs for better semantic coherence
                    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                    for idx, paragraph in enumerate(paragraphs):
                        chunk_id = f"{case_id}:{filename}:{page_number + 1}:{idx}"
                        metadata = {
                            "case_id": case_id,
                            "document": filename,
                            "page": page_number + 1,
                            "paragraph_index": idx,
                        }
                        # Compute embedding for the paragraph
                        embedding = self.embedding_fn([paragraph])[0]
                        # Upsert into the collection
                        self.collection.add(
                            documents=[paragraph],
                            metadatas=[metadata],
                            ids=[chunk_id],
                            embeddings=[embedding],
                        )

    def query(self, query_text: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Perform a semantic search over the ingested documents and return
        ``top_k`` most relevant chunks along with their metadata.

        Parameters
        ----------
        query_text: str
            The natural language query used to find relevant paragraphs.
        top_k: int
            How many results to return.  Defaults to 5.

        Returns
        -------
        List[DocumentChunk]
            A list of document chunks sorted by relevance.
        """
        results = self.collection.query(
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
                    page_number=metadata.get("page"),
                    text=doc,
                    metadata=metadata,
                )
            )
        return chunks
