"""
Legal Case Management AI System
--------------------------------

This script demonstrates how to build an end‑to‑end system that ingests legal
documents, stores their semantic representations in a vector database, exposes
database records via an MCP server, and generates a professional demand letter
using a large language model (LLM).

The code is organised into three primary components:

1. Document ingestion and retrieval (RAG)
2. MCP server implementation
3. Demand letter generation

Before running this script you must:
 - Install required dependencies (see requirements in README)
 - Configure your PostgreSQL connection URL
 - Provide an Azure OpenAI API key and deployment names for embeddings and LLM

This file is meant as a starting point and can be extended or refactored
according to the specific needs of your deployment.

Author: Your Name
Date: 2025-08-02
"""

import os
import json
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF for PDF processing
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlalchemy
from sqlalchemy import create_engine, text

import openai


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# PostgreSQL connection URL (e.g., "postgresql+psycopg2://user:password@localhost/db")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://user:password@localhost/legal_db")

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://your-resource-name.openai.azure.com")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")

# Directory containing case documents
CASE_DOCS_DIR = os.environ.get("CASE_DOCS_DIR", "sample_docs/2024-PI-001")


# Configure OpenAI client
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = "azure"
openai.api_version = "2023-05-15"  # adjust according to your deployment


# -----------------------------------------------------------------------------
# Utility classes and functions
# -----------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """Represents a chunk of text extracted from a document."""
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
    """

    def __init__(self, collection_name: str = "legal_docs") -> None:
        self.client = chromadb.Client(Settings(allow_reset=True))
        # Reset any existing collection to ensure idempotent runs
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(collection_name)
        self.collection = self.client.create_collection(collection_name)
        # Configure embedding function using Azure OpenAI
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=AZURE_OPENAI_API_KEY,
            api_base=AZURE_OPENAI_ENDPOINT,
            api_type="azure",
            api_version="2023-05-15",
            deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )

    def ingest_case_documents(self, case_id: str, docs_dir: str) -> None:
        """
        Ingest all PDF documents for a given case. This will extract text,
        split it into chunks, compute embeddings, and upsert them into Chroma.
        """
        for filename in os.listdir(docs_dir):
            if not filename.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(docs_dir, filename)
            print(f"Processing {file_path} …")
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

        print("Ingestion complete.")

    def query(self, query_text: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Perform a semantic search over the ingested documents and return the top_k
        most relevant chunks along with their metadata.
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
                    id=metadata.get("case_id") + ":" + metadata.get("document"),
                    case_id=metadata.get("case_id"),
                    doc_name=metadata.get("document"),
                    page_number=metadata.get("page"),
                    text=doc,
                    metadata=metadata,
                )
            )
        return chunks


class MCPDatabase:
    """
    Provides helper methods to interact with the PostgreSQL case management
    database. This class implements the MCP server functions defined in
    the technical specification.
    """

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.engine = create_engine(database_url)

    # Helper for executing SQL and returning a list of dictionaries
    def _execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_case_details(self, case_id: str) -> Dict[str, Any]:
        query = """
            SELECT case_id, case_type, date_filed, status, attorney_id, case_summary
            FROM cases
            WHERE case_id = :case_id
        """
        rows = self._execute(query, {"case_id": case_id})
        if not rows:
            raise HTTPException(status_code=404, detail="Case not found")
        return rows[0]

    def get_case_documents(self, case_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT doc_id, case_id, file_path, doc_category, upload_date, document_title, metadata
            FROM documents
            WHERE case_id = :case_id
        """
        params = {"case_id": case_id}
        if category:
            query += " AND doc_category = :category"
            params["category"] = category
        return self._execute(query, params)

    def get_case_timeline(self, case_id: str, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT event_id, case_id, event_date, event_type, description, amount
            FROM case_events
            WHERE case_id = :case_id
            ORDER BY event_date ASC
        """
        params = {"case_id": case_id}
        if event_type:
            query += " AND event_type = :event_type"
            params["event_type"] = event_type
        return self._execute(query, params)

    def get_financial_summary(self, case_id: str) -> Dict[str, float]:
        query = """
            SELECT
                SUM(CASE WHEN event_type = 'medical_treatment' THEN amount ELSE 0 END) AS total_medical,
                SUM(CASE WHEN event_type = 'expense' THEN amount ELSE 0 END) AS total_expenses,
                SUM(CASE WHEN event_type = 'lost_wage' THEN amount ELSE 0 END) AS total_lost_wages
            FROM case_events
            WHERE case_id = :case_id
        """
        rows = self._execute(query, {"case_id": case_id})
        return rows[0] if rows else {}

    def get_party_details(self, case_id: str, party_type: str) -> List[Dict[str, Any]]:
        query = """
            SELECT party_id, case_id, party_type, name, contact_info, insurance_info
            FROM parties
            WHERE case_id = :case_id AND party_type = :party_type
        """
        return self._execute(query, {"case_id": case_id, "party_type": party_type})

    def search_similar_cases(self, case_type: str, keywords: List[str]) -> List[Dict[str, Any]]:
        # A simple example of keyword search. In production you might
        # implement a full‑text search index or use vector search for precedents.
        keyword_filter = " OR ".join([f"case_summary ILIKE :kw{i}" for i in range(len(keywords))])
        params = {"case_type": case_type}
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = f"%{kw}%"
        query = f"""
            SELECT case_id, case_type, date_filed, status, case_summary
            FROM cases
            WHERE case_type = :case_type AND ({keyword_filter})
            ORDER BY date_filed DESC
            LIMIT 10
        """
        return self._execute(query, params)


# -----------------------------------------------------------------------------
# LLM‑based demand letter generation
# -----------------------------------------------------------------------------

class DemandLetterGenerator:
    """
    Generates a demand letter by combining structured case data and unstructured
    document snippets. Uses Azure OpenAI's GPT‑4 (or equivalent) for
    drafting the letter.
    """

    def __init__(self, mcp_db: MCPDatabase, doc_ingestor: DocumentIngestor) -> None:
        self.mcp_db = mcp_db
        self.doc_ingestor = doc_ingestor

    def _assemble_context(self, case_id: str, focus: str) -> str:
        """
        Fetch data from the MCP server and RAG system, then assemble a
        comprehensive prompt context for the LLM.
        """
        # 1. Case details and parties
        case_details = self.mcp_db.get_case_details(case_id)
        plaintiff = self.mcp_db.get_party_details(case_id, "plaintiff")[0]
        defendant = self.mcp_db.get_party_details(case_id, "defendant")[0]
        financials = self.mcp_db.get_financial_summary(case_id)

        # 2. Retrieve relevant document chunks
        rag_results = self.doc_ingestor.query(focus, top_k=5)

        # 3. Build narrative sections
        context_sections = []
        context_sections.append(f"Case ID: {case_details['case_id']}, Type: {case_details['case_type']}, "
                                f"Status: {case_details['status']}, Date filed: {case_details['date_filed']}\n")
        context_sections.append(f"Plaintiff: {plaintiff['name']}, Defendant: {defendant['name']}\n")
        context_sections.append("Key financials:\n"
                                f" - Total medical expenses: ${financials.get('total_medical', 0):,.2f}\n"
                                f" - Total lost wages: ${financials.get('total_lost_wages', 0):,.2f}\n"
                                f" - Other expenses: ${financials.get('total_expenses', 0):,.2f}\n")
        context_sections.append("Relevant document excerpts:")
        for chunk in rag_results:
            citation = f"[{chunk.doc_name}, p.{chunk.page_number}]"
            context_sections.append(f"{citation}: {chunk.text.strip()}\n")

        # Join all sections together
        return "\n".join(context_sections)

    def generate_demand_letter(self, case_id: str, focus: str) -> str:
        """
        Public method to generate the demand letter. It assembles the context
        and sends it to the OpenAI chat completion endpoint.
        """
        context = self._assemble_context(case_id, focus)
        system_prompt = (
            "You are an experienced personal injury attorney drafting a demand letter. "
            "Use the provided case details and relevant document excerpts to create a "
            "professional, persuasive demand letter. The letter should include: (1) a "
            "statement of facts; (2) liability discussion referencing the police report; "
            "(3) detailed damages (medical, lost wages, pain and suffering) with amounts; "
            "(4) conclusion and demand. Include proper citations to the source documents "
            "and maintain a respectful tone."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]
        # Call the OpenAI ChatCompletion API
        response = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_CHAT_DEPLOYMENT,
            model="gpt-4",
            messages=messages,
            max_tokens=1500,
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"].strip()


# -----------------------------------------------------------------------------
# FastAPI MCP server setup
# -----------------------------------------------------------------------------

app = FastAPI(title="Legal MCP Server")

# Instantiate database and document ingestor on startup
mcp_db = MCPDatabase()
doc_ingestor = DocumentIngestor()
demand_gen = DemandLetterGenerator(mcp_db, doc_ingestor)


# Pydantic models for API input
class DocumentIngestRequest(BaseModel):
    case_id: str
    docs_dir: str


class DemandLetterRequest(BaseModel):
    case_id: str
    focus: str  # e.g., "medical expenses and lost wages"


# Ingestion endpoint
@app.post("/ingest-documents")
def ingest_documents(req: DocumentIngestRequest) -> Dict[str, str]:
    doc_ingestor.ingest_case_documents(req.case_id, req.docs_dir)
    return {"status": "success"}


# MCP endpoints
@app.get("/case/{case_id}")
def get_case_details(case_id: str) -> Dict[str, Any]:
    return mcp_db.get_case_details(case_id)


@app.get("/case/{case_id}/documents")
def get_case_documents(case_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
    return mcp_db.get_case_documents(case_id, category)


@app.get("/case/{case_id}/timeline")
def get_case_timeline(case_id: str, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
    return mcp_db.get_case_timeline(case_id, event_type)


@app.get("/case/{case_id}/financials")
def get_financial_summary(case_id: str) -> Dict[str, float]:
    return mcp_db.get_financial_summary(case_id)


@app.get("/case/{case_id}/parties/{party_type}")
def get_party_details(case_id: str, party_type: str) -> List[Dict[str, Any]]:
    return mcp_db.get_party_details(case_id, party_type)


@app.post("/generate-demand-letter")
def generate_demand_letter(req: DemandLetterRequest) -> Dict[str, str]:
    """
    Generate a demand letter for the specified case and focus area. This
    endpoint will return the drafted letter as a string.
    """
    letter = demand_gen.generate_demand_letter(req.case_id, req.focus)
    return {"demand_letter": letter}


if __name__ == "__main__":
    import uvicorn
    # Optionally ingest documents on startup (for demonstration)
    if os.path.exists(CASE_DOCS_DIR):
        doc_ingestor.ingest_case_documents("2024-PI-001", CASE_DOCS_DIR)
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)