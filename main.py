"""
main.py
--------

This is the entry point for the Legal Case Management AI System.  It wires
together the RAG pipeline, the database access layer and the demand letter
generator into a FastAPI application.  It also optionally exposes a
Model Context Protocol (MCP) server when the ``mcp`` and ``mcp-sqlalchemy``
packages are installed and the script is invoked with the ``mcp``
command‑line argument.

Usage
-----

Run the FastAPI server (default):

    python main.py

This starts a web server on ``localhost:8000`` exposing the following
endpoints:

* ``POST /ingest-documents`` – Ingest PDF documents for a case.
* ``GET /case/{case_id}`` – Retrieve case details.
* ``GET /case/{case_id}/documents`` – List case documents, optionally
  filtering by category.
* ``GET /case/{case_id}/timeline`` – Retrieve timeline events.
* ``GET /case/{case_id}/financials`` – Get a financial summary.
* ``GET /case/{case_id}/parties/{party_type}`` – Retrieve party details.
* ``POST /generate-demand-letter`` – Generate a demand letter and return
  the text plus a path to the generated PDF.

Run the MCP server:

    python main.py mcp

This requires the optional ``mcp`` and ``mcp-sqlalchemy`` packages to
be installed.  See the README for installation instructions.  The MCP
server exposes database functions through a language model interface.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from db import DBAccessor
from rag_pipline import DocumentIngestor
from generate_letter import DemandLetterGenerator


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------

# Instantiate database accessor, document ingestor and letter generator
mcp_db = DBAccessor()

# Initialise the document ingestor.  This version uses OpenAI embeddings
doc_ingestor = DocumentIngestor()

# The demand letter generator relies on the OpenAI API via the
# ``OPENAI_API_KEY`` environment variable.  If no API key is provided
# the generation will fail at runtime.
demand_gen = DemandLetterGenerator(mcp_db, doc_ingestor)

# Create the FastAPI app
app = FastAPI(title="Legal MCP Server")


# ------------------------------------------------------------------
# Pydantic models for API input
# ------------------------------------------------------------------

class DocumentIngestRequest(BaseModel):
    case_id: str
    docs_dir: str


class DemandLetterRequest(BaseModel):
    case_id: str
    focus: str  # e.g., "medical expenses and lost wages"


# ------------------------------------------------------------------
# FastAPI route definitions
# ------------------------------------------------------------------

@app.post("/ingest-documents")
def ingest_documents(req: DocumentIngestRequest) -> Dict[str, str]:
    """
    Ingest PDF documents into the vector store for the given case.  This
    endpoint wraps the functionality of ``DocumentIngestor.ingest_case_documents``.
    Returns a simple status message on success.
    """
    try:
        doc_ingestor.ingest_case_documents(req.case_id, req.docs_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success"}


@app.get("/case/{case_id}")
def get_case_details(case_id: str) -> Dict[str, Any]:
    """Retrieve general information about a case."""
    data = mcp_db.get_case_details(case_id)
    if not data:
        raise HTTPException(status_code=404, detail="Case not found")
    return data


@app.get("/case/{case_id}/documents")
def get_case_documents(case_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List documents associated with the case.  Can filter by category."""
    return mcp_db.get_case_documents(case_id, category)


@app.get("/case/{case_id}/timeline")
def get_case_timeline(case_id: str, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve a chronological view of events for the case."""
    return mcp_db.get_case_timeline(case_id, event_type)


@app.get("/case/{case_id}/financials")
def get_financial_summary(case_id: str) -> Dict[str, float]:
    """Return a summary of medical expenses, lost wages and other costs."""
    return mcp_db.get_financial_summary(case_id)


@app.get("/case/{case_id}/parties/{party_type}")
def get_party_details(case_id: str, party_type: str) -> List[Dict[str, Any]]:
    """Retrieve specific party details (plaintiff, defendant, etc.)."""
    return mcp_db.get_party_details(case_id, party_type)


@app.post("/generate-demand-letter")
def generate_demand_letter(req: DemandLetterRequest) -> Dict[str, str]:
    """
    Generate a demand letter for the specified case and focus area.  This
    endpoint returns both the drafted letter text and the path to a PDF
    file saved on the server.  The PDF is generated using PyMuPDF via
    ``DemandLetterGenerator.save_letter_as_pdf``.
    """
    try:
        letter_text = demand_gen.generate_demand_letter_text(req.case_id, req.focus)
        # Determine an output directory for generated PDFs; fall back to current working directory
        output_dir = os.getenv("DEMAND_LETTER_OUTPUT_DIR", os.getcwd())
        os.makedirs(output_dir, exist_ok=True)
        pdf_filename = f"demand_letter_{req.case_id}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        demand_gen.save_letter_as_pdf(letter_text, pdf_path)
        return {"demand_letter": letter_text, "pdf_path": pdf_path}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ------------------------------------------------------------------
# MCP server integration (optional)
# ------------------------------------------------------------------

def run_mcp_server() -> None:
    """
    Launch the MCP server exposing the case management context.  This
    function is invoked when the script is run with the ``mcp`` argument.
    It expects the optional ``mcp`` and ``mcp-sqlalchemy`` libraries to
    be installed.  If they are not available a helpful error message is
    printed and the program exits.
    """
    try:
        from mcp import ContextProvider
        from mcp import MCPServer
        from mcp import tools
    except ImportError:
        print(
            "Error: The 'mcp' and 'mcp-sqlalchemy' packages are required to run the MCP server.\n"
            "Install them with 'pip install mcp mcp-sqlalchemy' and try again."
        )
        sys.exit(1)

    class CaseContextProvider(ContextProvider):
        """
        Implements the MCP context provider interface by delegating calls
        to the ``DBAccessor`` methods.  This class is registered with
        ``MCPServer`` to expose database functions to language models.
        """

        def __init__(self) -> None:
            self.db = mcp_db

        def get_case_details(self, case_id: str) -> Dict[str, Any]:
            return self.db.get_case_details(case_id) or {}

        def get_case_documents(self, case_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
            return self.db.get_case_documents(case_id, category)

        def get_case_timeline(self, case_id: str, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
            return self.db.get_case_timeline(case_id, event_type)

        def get_financial_summary(self, case_id: str) -> Dict[str, Any]:
            return self.db.get_financial_summary(case_id)

        def search_similar_cases(self, case_type: str, keywords: List[str]) -> List[Dict[str, Any]]:
            return self.db.search_similar_cases(case_type, keywords)

        def get_party_details(self, case_id: str, party_type: str) -> List[Dict[str, Any]]:
            return self.db.get_party_details(case_id, party_type)

    # Instantiate the MCP server with the context provider and a simple echo tool
    server = MCPServer(context_providers=[CaseContextProvider()], tools=[tools.echo])
    server.run()


if __name__ == "__main__":
    # If the script is invoked with the 'mcp' argument run the MCP server
    if len(sys.argv) > 1 and sys.argv[1].lower() == "mcp":
        run_mcp_server()
    else:
        # Optionally ingest documents on startup if the CASE_DOCS_DIR environment variable is set
        case_docs_dir = os.getenv("CASE_DOCS_DIR")
        case_id_env = os.getenv("DEFAULT_CASE_ID")  # Optional environment variable to specify case ID on startup
        if case_docs_dir and case_id_env and os.path.isdir(case_docs_dir):
            try:
                doc_ingestor.ingest_case_documents(case_id_env, case_docs_dir)
            except Exception:
                # Swallow errors so server can still start
                pass
        # Start the FastAPI web server
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
