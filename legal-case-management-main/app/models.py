from sqlalchemy import Column, Integer, String, Date, Text, ForeignKey
from .db import Base

class Case(Base):
    __tablename__ = "cases"
    case_id = Column(String, primary_key=True, index=True)
    case_type = Column(String)
    date_filed = Column(Date)
    status = Column(String)
    attorney_id = Column(Integer)
    case_summary = Column(Text)

class Party(Base):
    __tablename__ = "parties"
    party_id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    party_type = Column(String)
    name = Column(String)
    contact_info = Column(String)

class TimelineEvent(Base):
    __tablename__ = "timeline_events"
    event_id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    event_date = Column(Date)
    description = Column(Text)

class FinancialRecord(Base):
    __tablename__ = "financial_records"
    record_id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    record_type = Column(String)
    amount = Column(Integer)
    description = Column(Text)
