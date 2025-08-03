from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, db

models.Base.metadata.create_all(bind=db.engine)

app = FastAPI()

def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

@app.get("/cases")
def get_cases(db: Session = Depends(get_db)):
    return db.query(models.Case).all()

@app.get("/parties")
def get_parties(db: Session = Depends(get_db)):
    return db.query(models.Party).all()

@app.get("/events")
def get_events(db: Session = Depends(get_db)):
    return db.query(models.TimelineEvent).all()

@app.get("/financials")
def get_financials(db: Session = Depends(get_db)):
    return db.query(models.FinancialRecord).all()
