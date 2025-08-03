# Legal AI Case Management System

## Overview
This project implements an AI-powered legal assistant that:
- Ingests structured medical/legal PDFs
- Uses a Retrieval-Augmented Generation (RAG) pipeline with legal-specific chunking
- Embeds document chunks using `OpenAIEmbeddings` and indexes with `ChromaDB`
- Queries those chunks based on natural-language questions
- Generates demand letters using Jinja2 templates
- Serves the solution via an MCP server built with FastAPI

## Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional but supported)
- OpenAI API Key (for embeddings + LLM)

## Setup
1. Start services (if using Docker):
   ```bash
   docker-compose up -d
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ingest PDFs and build vector store:
   ```bash
   python app/rag_pipeline.py
   ```

## Running

- Launch the MCP API server:
  ```bash
  uvicorn app.main:app --reload
  ```

- Generate a demand letter:
  ```bash
  python generate_letter.py
  ```
  Output saved to `generated_letter.txt`.

## RAG Workflow

1. Chunk PDFs using `PyMuPDF`
2. Embed text using `OpenAIEmbeddings` (LangChain)
3. Store vectors in `ChromaDB`
4. Query using natural language
5. Retrieve top-k chunks
6. Use Jinja2 to fill demand letter template with contextual info

---

## Dependencies

- `PyMuPDF` – Parse PDF files
- `openai`, `langchain` – Embedding & LLM pipeline
- `chromadb` – Vector storage
- `jinja2` – Demand letter templating
- `fastapi`, `uvicorn` – MCP REST API server

---

## Repo Structure

```
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── db.py
│   ├── models.py
├── templates/
│   └── demand_letter.jinja2
├── generate_letter.py
├── requirements.txt
├── README.md
```

---

## ✅ Deliverables

- [x] ✅ End-to-end pipeline (PDF → demand letter)
- [x] ✅ RAG-based retrieval with chunking & embeddings
- [x] ✅ MCP server via FastAPI
- [x] ✅ Sample demand letter generated
- [x] ✅ This documentation with architecture & instructions

---

For any questions, feel free to reach out at: **ramanadata568@gmail.com**
