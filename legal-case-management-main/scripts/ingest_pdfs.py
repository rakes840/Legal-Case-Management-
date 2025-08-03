import sys
from app.rag_pipeline import ingest_pdfs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_pdfs.py <case_directory>")
        sys.exit(1)
    case_dir = sys.argv[1]
    ingest_pdfs(case_dir)
    print(f"Ingested PDFs from {case_dir}")
