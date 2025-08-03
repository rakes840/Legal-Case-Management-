import requests
import os
from jinja2 import Environment, FileSystemLoader
from app.rag_pipeline import generate_response

# Configuration
MCP_API_BASE = "http://localhost:8000"
TEMPLATE_PATH = "templates"
OUTPUT_PATH = "generated_letter.txt"

# Step 1: Get case details from MCP server
def get_case_data(case_id):
    case = requests.get(f"{MCP_API_BASE}/cases").json()
    parties = requests.get(f"{MCP_API_BASE}/parties").json()
    events = requests.get(f"{MCP_API_BASE}/events").json()
    financials = requests.get(f"{MCP_API_BASE}/financials").json()

    # Filter everything by case_id
    case_data = next((c for c in case if c["case_id"] == case_id), {})
    party_data = [p for p in parties if p["case_id"] == case_id]
    event_data = [e for e in events if e["case_id"] == case_id]
    financial_data = [f for f in financials if f["case_id"] == case_id]

    return {
        "case": case_data,
        "parties": party_data,
        "events": event_data,
        "financials": financial_data,
    }

# Step 2: Get RAG facts
def get_context_from_rag(query):
    return generate_response(query)

# Step 3: Render final letter using Jinja2
def render_letter(data, rag_context):
    env = Environment(loader=FileSystemLoader(TEMPLATE_PATH))
    template = env.get_template("demand_letter.jinja2")

    rendered = template.render(
        case=data["case"],
        parties=data["parties"],
        events=data["events"],
        financials=data["financials"],
        context=rag_context,
    )

    with open(OUTPUT_PATH, "w") as f:
        f.write(rendered)
    print(f"✅ Demand letter written to {OUTPUT_PATH}")

if __name__ == "__main__":
    case_id = "2024-PI-001"
    print("📦 Fetching case data from MCP...")
    data = get_case_data(case_id)
    print("🧠 Querying RAG for medical expenses and lost wages...")
    context = get_context_from_rag("Summarize medical expenses and lost wages for case 2024-PI-001")
    print("📝 Rendering final letter...")
    render_letter(data, context)
