# app.py
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Local imports
from src.load_data import load_pdf, filter_minimal_docs, text_split
from src.embeddings import download_embeddings
from src.vector_store import initialize_pinecone, load_existing_vectorstore, create_vectorstore
from src.chatbot import create_chat_model, build_rag_chain

# Initialize FastAPI
app = FastAPI(title="Medical Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_DIR = "data"
INDEX_NAME = "medical-chatbot"
EMBEDDING_DIMENSION = 384  # For all-MiniLM-L6-v2

# Globals to persist across requests
embedding = None
docsearch = None
retriever = None
rag_chain = None

# Pydantic model for input
class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    global embedding, docsearch, retriever, rag_chain

    print("🚀 Starting up Medical Chatbot API...")

    # Validate env
    if not PINECONE_API_KEY or not GROQ_API_KEY:
        raise RuntimeError("❌ Missing required environment variables.")

    # Step 1: Load and split PDFs
    print("📄 Loading PDFs...")
    extracted_data = load_pdf(DATA_DIR)
    print(f"✅ Loaded {len(extracted_data)} pages")

    minimal_docs = filter_minimal_docs(extracted_data)
    texts_chunk = text_split(minimal_docs)
    print(f"✅ Split into {len(texts_chunk)} chunks")

    # Step 2: Load embeddings
    print("🧠 Loading embedding model...")
    embedding = download_embeddings()

    # Step 3: Initialize Pinecone
    print("🧱 Initializing Pinecone...")
    _, _ = initialize_pinecone(PINECONE_API_KEY, INDEX_NAME)

    # Step 4: Load or create vector store
    try:
        docsearch = load_existing_vectorstore(embedding, INDEX_NAME)
        print("✅ Loaded existing Pinecone index.")
    except Exception as e:
        print(f"⚠️ Failed to load existing index: {e}")
        print("📤 Creating and uploading to new index...")
        docsearch = create_vectorstore(texts_chunk, embedding, INDEX_NAME)

    # Step 5: Create retriever
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Step 6: Build RAG pipeline
    print("💬 Initializing chat model...")
    chat_model = create_chat_model(GROQ_API_KEY)
    rag_chain = build_rag_chain(chat_model, retriever)

    print("✅ Startup complete!")


@app.get("/")
def read_root():
    return {"message": "Medical Chatbot API is running."}


@app.post("/ask/")
def ask_question(payload: QuestionRequest):
    global rag_chain

    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Model not initialized yet.")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = rag_chain.invoke({"input": question})
        answer = response.get("answer", "Sorry, I don't know the answer.")
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")


@app.get("/questions/")
def ask_questions(questions: List[str] = Query(..., description="List of questions to ask")):
    global rag_chain

    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Model not initialized yet.")

    if not questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty.")

    results = []
    for q in questions:
        try:
            response = rag_chain.invoke({"input": q})
            answer = response.get("answer", "Sorry, I don't know the answer.")
            results.append({"question": q, "answer": answer})
        except Exception as e:
            results.append({"question": q, "answer": f"Error: {str(e)}"})

    return results


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pinecone_index": INDEX_NAME,
        "model_ready": rag_chain is not None,
    }
