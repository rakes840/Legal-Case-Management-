"""
generate_letter.py
------------------

This module defines the logic for generating professional demand letters
by combining structured data from the case management database with
unstructured snippets retrieved via RAG.  It leverages Azure OpenAI's
ChatCompletion API to draft the narrative and uses PyMuPDF to save the
resulting text as a PDF document.

The primary class exported by this module is ``DemandLetterGenerator``.
Create an instance by passing an already initialised ``DBAccessor`` and
``DocumentIngestor``.  Then call ``generate_demand_letter_text`` to
obtain the raw letter or ``generate_and_save_pdf`` to write it to disk.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import fitz  # PyMuPDF for PDF generation
import openai

from db import DBAccessor
from rag_pipline import DocumentIngestor, DocumentChunk


class DemandLetterGenerator:
    """
    Generates a demand letter by combining structured case data and
    unstructured document snippets.  Uses Azure OpenAI's GPT‑4 (or
    equivalent) for drafting the letter and PyMuPDF to render a PDF.
    """

    def __init__(self, mcp_db: DBAccessor, doc_ingestor: DocumentIngestor) -> None:
        self.mcp_db = mcp_db
        self.doc_ingestor = doc_ingestor
        # Configure OpenAI client using environment variables
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        openai.api_type = "azure"
        # Set a default API version; you may adjust this as required
        openai.api_version = "2023-05-15"
        self.chat_deployment = os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _assemble_context(self, case_id: str, focus: str) -> str:
        """
        Fetch data from the MCP database and RAG system, then assemble a
        comprehensive prompt context for the LLM.

        The context includes general case details, party information,
        aggregate financials, and excerpts from relevant documents.
        """
        # 1. Case details and parties
        case_details = self.mcp_db.get_case_details(case_id)
        if not case_details:
            raise ValueError(f"Case not found: {case_id}")

        # Fetch parties; fallback to placeholders if missing
        plaintiffs = self.mcp_db.get_party_details(case_id, "plaintiff")
        defendants = self.mcp_db.get_party_details(case_id, "defendant")
        plaintiff_name = plaintiffs[0]["name"] if plaintiffs else "Plaintiff"
        defendant_name = defendants[0]["name"] if defendants else "Defendant"

        financials = self.mcp_db.get_financial_summary(case_id)

        # 2. Retrieve relevant document chunks via RAG
        rag_results: List[DocumentChunk] = self.doc_ingestor.query(focus, top_k=5)

        # 3. Build narrative sections
        context_sections: List[str] = []
        context_sections.append(
            f"Case ID: {case_details.get('case_id')}, Type: {case_details.get('case_type')}, "
            f"Status: {case_details.get('status')}, Date filed: {case_details.get('date_filed')}\n"
        )
        context_sections.append(
            f"Plaintiff: {plaintiff_name}, Defendant: {defendant_name}\n"
        )
        context_sections.append(
            "Key financials:\n"
            f" - Total medical expenses: ${financials.get('total_medical', 0) or 0:,.2f}\n"
            f" - Total lost wages: ${financials.get('total_lost_wages', 0) or 0:,.2f}\n"
            f" - Other expenses: ${financials.get('total_expenses', 0) or 0:,.2f}\n"
        )
        context_sections.append("Relevant document excerpts:")
        for chunk in rag_results:
            citation = f"[{chunk.doc_name}, p.{chunk.page_number}]"
            context_sections.append(f"{citation}: {chunk.text.strip()}\n")

        # Join all sections together
        return "\n".join(context_sections)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_demand_letter_text(self, case_id: str, focus: str) -> str:
        """
        Generate the demand letter as raw text.  This method assembles the
        context and sends it to the OpenAI ChatCompletion API.  The result
        is the drafted letter as a string.

        Parameters
        ----------
        case_id: str
            Identifier of the case for which to generate the letter.
        focus: str
            A description of what aspects to emphasise (e.g., "medical
            expenses and lost wages").  Used as the query for the RAG system.

        Returns
        -------
        str
            The generated demand letter text.
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
        # Call the OpenAI ChatCompletion API; this invocation depends on Azure
        # deployment names specified in environment variables.  The ``chat_deployment``
        # attribute defaults to ``gpt-4`` if not provided.
        response = openai.ChatCompletion.create(
            deployment_id=self.chat_deployment,
            model="gpt-4",
            messages=messages,
            max_tokens=1500,
            temperature=0.2,
        )
        # Extract the content from the first choice
        return response["choices"][0]["message"]["content"].strip()

    def save_letter_as_pdf(self, letter_text: str, output_path: str) -> None:
        """
        Save a string of text as a PDF file.  This helper uses PyMuPDF to
        render the letter on a single page.  For long letters PyMuPDF
        automatically handles pagination.

        Parameters
        ----------
        letter_text: str
            The content to write to the PDF.
        output_path: str
            File path where the PDF should be saved.  Existing files are
            overwritten.
        """
        # Create a new blank PDF document
        doc = fitz.open()
        page = doc.new_page()
        # Define a rectangular region leaving standard margins (72pt = 1 inch)
        rect = fitz.Rect(72, 72, 540, 720)
        # Use a monospaced or proportional font; defaults are fine
        # Insert the text within the rectangle; text will wrap automatically
        page.insert_textbox(
            rect,
            letter_text,
            fontsize=12,
            fontname="Times-Roman",
            align=0,  # left align
        )
        # Write the document to disk
        doc.save(output_path)
        doc.close()

    def generate_and_save_pdf(
        self, case_id: str, focus: str, output_path: str
    ) -> str:
        """
        Convenience method that generates a demand letter and writes it to
        ``output_path`` as a PDF.  Returns the generated text for further
        processing.
        """
        letter_text = self.generate_demand_letter_text(case_id, focus)
        self.save_letter_as_pdf(letter_text, output_path)
        return letter_text
