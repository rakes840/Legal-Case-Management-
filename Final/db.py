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

try:
    # SQLAlchemy is optional.  If it is not installed, DBAccessor will fall back
    # to using an in‑memory sample dataset defined below.  Users who wish to
    # connect to a real database must install SQLAlchemy (e.g. via
    # ``pip install sqlalchemy psycopg2-binary``) and configure the
    # ``DATABASE_URL`` environment variable.
    from sqlalchemy import create_engine, text  # type: ignore
    from sqlalchemy.engine import Engine  # type: ignore
except ImportError:
    create_engine = None  # type: ignore
    text = None  # type: ignore
    Engine = None  # type: ignore


# PostgreSQL connection URL (can be overridden via the DATABASE_URL environment variable)
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://user:password@localhost/legal_db",
)


class DBAccessor:
    """
    Provides helper methods to interact with the case management database.

    If SQLAlchemy is available this class will attempt to connect to the
    configured database and execute queries.  If SQLAlchemy is not
    installed, the accessor falls back to an in‑memory dataset.  The
    fallback is intended for demonstration and testing purposes only.
    """

    # A minimal in‑memory representation of case data used when SQLAlchemy
    # is unavailable.  You can extend this dictionary or load it from
    # external JSON/YAML if needed.  The structure mirrors the expected
    # database tables.
    _SAMPLE_DATA: Dict[str, Any] = {
        "cases": [
            {
                "case_id": "2024-PI-001",
                "case_type": "Personal Injury - Motor Vehicle Accident",
                "date_filed": "2024-03-15",
                "status": "Active - Demand Phase",
                "attorney_id": 1,
                "case_summary": (
                    "Plaintiff was injured in a motor vehicle accident and seeks damages "
                    "for medical expenses, lost wages and pain and suffering."
                ),
            }
        ],
        "parties": [
            {
                "party_id": 1,
                "case_id": "2024-PI-001",
                "party_type": "plaintiff",
                "name": "John Smith",
                "contact_info": {},
                "insurance_info": {},
            },
            {
                "party_id": 2,
                "case_id": "2024-PI-001",
                "party_type": "defendant",
                "name": "ABC Insurance Company",
                "contact_info": {},
                "insurance_info": {},
            },
        ],
        "case_events": [],  # populate if needed
        "documents": [],
    }

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        # If SQLAlchemy is available and create_engine is defined, connect to DB
        if create_engine is not None:
            self.engine: Optional[Engine] = create_engine(database_url, future=True)  # type: ignore
        else:
            self.engine = None

    def _execute(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Helper for executing SQL and returning a list of dictionaries.  This
        method hides the boilerplate of opening and closing connections.
        """
        # If there is no database engine (SQLAlchemy not installed), return an empty
        # list.  Callers should handle this gracefully or rely on the fallback
        # in‑memory data defined above.
        if self.engine is None:
            return []
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})  # type: ignore
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]

    # ------------------------------------------------------------------
    # Case retrieval methods
    # ------------------------------------------------------------------
    def get_case_details(self, case_id: str) -> Dict[str, Any]:
        """Retrieve complete case information including general details."""
        if self.engine is None:
            # Fallback to sample data
            for case in self._SAMPLE_DATA.get("cases", []):
                if case["case_id"] == case_id:
                    return case
            return {}
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
        if self.engine is None:
            # Fallback: return any matching documents from sample data
            docs = [d for d in self._SAMPLE_DATA.get("documents", []) if d["case_id"] == case_id]
            if category:
                docs = [d for d in docs if d.get("doc_category") == category]
            return docs
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
        if self.engine is None:
            events = [e for e in self._SAMPLE_DATA.get("case_events", []) if e["case_id"] == case_id]
            if event_type:
                events = [e for e in events if e.get("event_type") == event_type]
            # Sort by date if available
            return sorted(events, key=lambda x: x.get("event_date", ""))
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
        if self.engine is None:
            # Aggregate from sample data
            total_medical = 0.0
            total_expenses = 0.0
            total_lost_wages = 0.0
            for event in self._SAMPLE_DATA.get("case_events", []):
                if event.get("case_id") != case_id:
                    continue
                if event.get("event_type") == "medical_treatment":
                    total_medical += float(event.get("amount", 0))
                elif event.get("event_type") == "expense":
                    total_expenses += float(event.get("amount", 0))
                elif event.get("event_type") == "lost_wage":
                    total_lost_wages += float(event.get("amount", 0))
            return {
                "total_medical": total_medical,
                "total_expenses": total_expenses,
                "total_lost_wages": total_lost_wages,
            }
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
        if self.engine is None:
            # Fallback: simple substring search on sample data
            results = []
            for case in self._SAMPLE_DATA.get("cases", []):
                if case.get("case_type") != case_type:
                    continue
                summary = case.get("case_summary", "").lower()
                if any(kw.lower() in summary for kw in keywords):
                    results.append(case)
            return results[:10]
        return self._execute(query, params)
