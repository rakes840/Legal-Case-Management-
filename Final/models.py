"""
models.py
-----------

This module defines SQLAlchemy ORM models representing the core tables used
by the Legal Case Management AI System.  The schema mirrors the database
definition outlined in the technical challenge specification.  If you need
to extend or modify the schema (for example to add indexes or constraints),
you should do so here.

The ``Base`` class is exported so that other modules (such as Alembic
migrations or the MCP context provider) can access the declarative metadata.
"""

from __future__ import annotations

import os
from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    Text,
    Float,
    ForeignKey,
    Numeric,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship


# Base class for all ORM models
Base = declarative_base()


class Case(Base):
    __tablename__ = "cases"

    case_id = Column(String(50), primary_key=True)
    case_type = Column(String(100))
    date_filed = Column(Date)
    status = Column(String(50))
    attorney_id = Column(Integer)
    case_summary = Column(Text)

    # Relationships
    parties = relationship("Party", back_populates="case", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="case", cascade="all, delete-orphan")
    events = relationship("CaseEvent", back_populates="case", cascade="all, delete-orphan")


class Party(Base):
    __tablename__ = "parties"

    party_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(50), ForeignKey("cases.case_id"), index=True)
    party_type = Column(String(50))  # 'plaintiff', 'defendant', 'witness', 'insurance_company'
    name = Column(String(200))
    contact_info = Column(JSON)
    insurance_info = Column(JSON)

    # Relationship
    case = relationship("Case", back_populates="parties")


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(50), ForeignKey("cases.case_id"), index=True)
    file_path = Column(String(500))
    doc_category = Column(String(100))  # 'medical', 'financial', 'correspondence', 'police_report'
    upload_date = Column(Date)
    document_title = Column(String(300))
    metadata = Column(JSON)

    # Relationship
    case = relationship("Case", back_populates="documents")


class CaseEvent(Base):
    __tablename__ = "case_events"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(50), ForeignKey("cases.case_id"), index=True)
    event_date = Column(Date)
    event_type = Column(String(100))  # 'accident', 'medical_treatment', 'expense', 'correspondence'
    description = Column(Text)
    amount = Column(Numeric(10, 2))  # for expenses, damages, bills

    # Relationship
    case = relationship("Case", back_populates="events")
