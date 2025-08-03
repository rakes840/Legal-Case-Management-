import os
import fitz  # PyMuPDF
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

CHROMA_DIR = "rag_store"
PDF_DIR = "sample_docs"

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdfs(pdf_dir):
    docs = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                text = ""
                with fitz.open(full_path) as doc:
                    for page in doc:
                        text += page.get_text()
                docs.append(Document(page_content=text, metadata={"source": file}))
    return docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

def generate_response(query):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=retriever)
    return qa.run(query)

if __name__ == "__main__":
    print("📄 Extracting and chunking PDFs...")
    docs = extract_text_from_pdfs(PDF_DIR)
    chunks = chunk_documents(docs)
    print("📥 Building vector store...")
    build_vectorstore(chunks)
    print("✅ RAG store is ready. Try calling generate_response() with your query.")
