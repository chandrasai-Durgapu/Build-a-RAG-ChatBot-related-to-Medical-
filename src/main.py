# main.py
import os
from dotenv import load_dotenv

from src.load_data import load_pdf, filter_minimal_docs, text_split
from src.embeddings import download_embeddings
from src.vector_store import initialize_pinecone, create_vectorstore
from src.chatbot import create_chat_model, build_rag_chain, answer_questions



def main():
    print("Medical Chatbot")

    # Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Load and preprocess documents
    extracted_data = load_pdf("data")
    print(f"Loaded {len(extracted_data)} pages from PDFs.")

    minimal_docs = filter_minimal_docs(extracted_data)
    texts_chunk = text_split(minimal_docs)
    print(f"Number of chunks created: {len(texts_chunk)}")

    # Embeddings
    embedding = download_embeddings()

    # Pinecone init and vectorstore creation
    index_name = "medical-chatbot"
    pc, index = initialize_pinecone(PINECONE_API_KEY, index_name)
    docsearch = create_vectorstore(texts_chunk, embedding, index_name)

    # Retriever
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Chat model and RAG chain
    chatModel = create_chat_model(GROQ_API_KEY)
    rag_chain = build_rag_chain(chatModel, retriever)

    # Sample questions
    questions = [
        "what is Acromegaly and gigantism?",
        "what is Acne?",
        "what is the Treatment of Acne?",
        "who is chandra sai?"
    ]

    answer_questions(rag_chain, questions)


if __name__ == "__main__":
    main()
