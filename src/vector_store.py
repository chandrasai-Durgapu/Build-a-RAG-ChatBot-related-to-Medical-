from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import os

from pinecone import Pinecone

def initialize_pinecone(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)

    # Get or create the index
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(name=index_name, metric="cosine")  # update dimension if needed

    index = pc.Index(index_name)
    return pc, index

def create_vectorstore(texts_chunk: list[Document], embedding: Embeddings, index_name: str) -> PineconeVectorStore:
    """
    Upload documents to an existing Pinecone index.
    Assumes the index is already created manually in Pinecone dashboard or via REST API.
    """
    print(f"Uploading {len(texts_chunk)} documents to Pinecone index '{index_name}'...")
    docsearch = PineconeVectorStore.from_documents(
        texts_chunk,
        embedding,
        index_name=index_name
    )
    return docsearch


def load_existing_vectorstore(embedding: Embeddings, index_name: str) -> PineconeVectorStore:
    """
    Load an existing Pinecone index as a vector store.
    """
    print(f"Loading existing Pinecone index '{index_name}'...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )
    return docsearch
