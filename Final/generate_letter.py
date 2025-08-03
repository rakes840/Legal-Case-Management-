"""
generate_letter.py
------------------

This module defines the logic for generating professional demand letters
by combining structured data from the case management database with
unstructured snippets retrieved via a retrieval‑augmented generation
pipeline.  It uses the OpenAI ChatCompletion API (non‑Azure) to draft
the narrative and leverages Matplotlib to render the resulting text to
a PDF.  The dependence on external libraries (such as PyMuPDF) has
been removed to make the code simpler to run in constrained
environments.

The primary class exported by this module is ``DemandLetterGenerator``.
Create an instance by passing an already initialised ``DBAccessor`` and
``DocumentIngestor``.  Then call ``generate_demand_letter_text`` to
obtain the raw letter or ``generate_and_save_pdf`` to write it to disk.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

try:
    import openai  # type: ignore
except ImportError:
    # ``openai`` is an optional dependency.  It must be installed by the
    # user to enable letter generation.  We import lazily within
    # ``DemandLetterGenerator``.
    openai = None  # type: ignore

import matplotlib.pyplot as plt
import textwrap

from db import DBAccessor
from rag_pipline import DocumentIngestor, DocumentChunk


class DemandLetterGenerator:
    """
    Generates a demand letter by combining structured case data and
    unstructured document snippets.  Uses the OpenAI ChatCompletion API
    to draft the letter and Matplotlib to render it to a PDF.  The
    letter generator itself is agnostic to the specific vector store
    implementation: it relies on a provided ``DBAccessor`` (for
    structured data) and ``DocumentIngestor`` (for RAG results).  To
    use this class you must set the ``OPENAI_API_KEY`` environment
    variable with a valid OpenAI API key.
    """

    def __init__(self, mcp_db: DBAccessor, doc_ingestor: DocumentIngestor) -> None:
        """
        Initialize the demand letter generator.  This class combines
        structured data from the ``mcp_db`` with unstructured excerpts from
        ``doc_ingestor`` and uses the OpenAI ChatCompletion API to draft
        the letter.  Set the ``OPENAI_API_KEY`` environment variable
        before instantiating this class.  You may override the chat model
        via ``OPENAI_CHAT_MODEL`` (defaults to ``gpt-4").
        """
        self.mcp_db = mcp_db
        self.doc_ingestor = doc_ingestor
        # Import openai lazily so the module can be imported on systems
        # without the openai package installed.  Generation will fail at
        # runtime if openai is unavailable.
        if openai is None:
            raise ImportError(
                "The 'openai' package is required for demand letter generation. "
                "Please install it with `pip install openai` and set the OPENAI_API_KEY environment variable."
            )
        # Store a reference to the openai module for later use
        self.openai = openai  # type: ignore
        # Configure OpenAI client using environment variables
        self.openai.api_key = os.getenv("OPENAI_API_KEY", "")  # type: ignore
        # Choose the chat model; default to gpt‑4
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")

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
        response = self.openai.ChatCompletion.create(  # type: ignore
            model=self.chat_model,
            messages=messages,
            max_tokens=1500,
            temperature=0.2,
        )  # type: ignore
        # Extract the content from the first choice
        return response["choices"][0]["message"]["content"].strip()

    def save_letter_as_pdf(self, letter_text: str, output_path: str) -> None:
        """
        Save a string of text as a PDF file.  This helper uses Matplotlib to
        render the letter onto a PDF page.  The text is wrapped to fit
        within an 8.5x11 inch page with 1‑inch margins.  This approach
        avoids the need for external PDF libraries (such as PyMuPDF) that
        may not be installed by default.

        Parameters
        ----------
        letter_text: str
            The content to write to the PDF.
        output_path: str
            File path where the PDF should be saved.  Existing files are
            overwritten.
        """
        # Create a figure sized for US Letter (8.5 x 11 inches)
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        # Wrap the letter text to fit within the page margins.  We assume
        # approximately 90 characters per line at a font size of 12.
        wrapped = textwrap.wrap(letter_text, width=90)
        y_pos = 1.0  # normalized coordinate for top of page
        line_height = 0.023  # approximate line height in figure coordinates
        for line in wrapped:
            ax.text(0.05, y_pos, line, fontsize=12, family='serif', transform=ax.transAxes)
            y_pos -= line_height
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

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
