"""
db.py
------

This module encapsulates interaction with the relational database used by the
Legal Case Management AI System.  It exposes a ``DBAccessor`` class that
provides methods for common operations such as retrieving case details,
document metadata, timeline events, financial summaries and party
information.  The methods in this class correspond to the required MCP
functions described in the technical challenge.

Internally, ``DBAccessor`` uses SQLAlchemy's ``create_engine`` and
``text`` constructs to execute parametrised SQL queries.  If you prefer to
use SQLAlchemy ORM queries instead of raw SQL, you could easily modify
these methods to work with the models defined in ``models.py``.

Configure the ``DATABASE_URL`` environment variable to point to your
PostgreSQL (or other supported) database before instantiating
``DBAccessor``.  By default it connects to ``postgresql+psycopg2://user:password@localhost/legal_db``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# PostgreSQL connection URL (can be overridden via the DATABASE_URL environment variable)
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://user:password@localhost/legal_db",
)


class DBAccessor:
    """
    Provides helper methods to interact with the case management database.

    This class implements the MCP server functions defined in the technical
    specification: ``get_case_details``, ``get_case_documents``,
    ``get_case_timeline``, ``get_financial_summary``,
    ``search_similar_cases`` and ``get_party_details``.
    """

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        # create_engine accepts URLs in the form dialect+driver://user:pass@host/db
        self.engine: Engine = create_engine(database_url, future=True)

    def _execute(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Helper for executing SQL and returning a list of dictionaries.  This
        method hides the boilerplate of opening and closing connections.
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]

    # ------------------------------------------------------------------
    # Case retrieval methods
    # ------------------------------------------------------------------
    def get_case_details(self, case_id: str) -> Dict[str, Any]:
        """Retrieve complete case information including general details."""
        query = """
            SELECT case_id, case_type, date_filed, status, attorney_id, case_summary
            FROM cases
            WHERE case_id = :case_id
        """
        rows = self._execute(query, {"case_id": case_id})
        return rows[0] if rows else {}

    def get_case_documents(
        self, case_id: str, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get document file paths and metadata, optionally filtered by category."""
        query = """
            SELECT doc_id, case_id, file_path, doc_category, upload_date, document_title, metadata
            FROM documents
            WHERE case_id = :case_id
        """
        params: Dict[str, Any] = {"case_id": case_id}
        if category:
            query += " AND doc_category = :category"
            params["category"] = category
        return self._execute(query, params)

    def get_case_timeline(
        self, case_id: str, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve chronological events for the case."""
        query = """
            SELECT event_id, case_id, event_date, event_type, description, amount
            FROM case_events
            WHERE case_id = :case_id
        """
        params: Dict[str, Any] = {"case_id": case_id}
        if event_type:
            query += " AND event_type = :event_type"
            params["event_type"] = event_type
        query += " ORDER BY event_date ASC"
        return self._execute(query, params)

    def get_financial_summary(self, case_id: str) -> Dict[str, float]:
        """
        Calculate total medical expenses, lost wages, and other damages for a case.

        This implementation relies on the ``case_events`` table where the
        ``event_type`` column distinguishes between different categories.
        Modify the SQL below if your schema differs.
        """
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
        """Get specific party information (plaintiff, defendant, etc.)."""
        query = """
            SELECT party_id, case_id, party_type, name, contact_info, insurance_info
            FROM parties
            WHERE case_id = :case_id AND party_type = :party_type
        """
        return self._execute(query, {"case_id": case_id, "party_type": party_type})

    def search_similar_cases(self, case_type: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Find precedent cases for reference.

        This method performs a simple keyword search against the ``case_summary``
        field.  In production you might implement a full‑text search or use
        vector search for precedents.
        """
        # Build a filter expression that searches for any of the keywords
        keyword_filter = " OR ".join([
            f"case_summary ILIKE :kw{i}" for i in range(len(keywords))
        ])
        params: Dict[str, Any] = {"case_type": case_type}
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
