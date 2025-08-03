from pydantic import BaseModel
from datetime import date
from typing import List

class PartyOut(BaseModel):
    name: str
    role: str

class EventOut(BaseModel):
    event_type: str
    description: str
    amount: float
    event_date: date

class CaseDetails(BaseModel):
    case: dict
    parties: List[PartyOut]
    events: List[EventOut]

    class Config:
        orm_mode = True
