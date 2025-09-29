# embeddings.py


# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def download_embeddings():
    
    """Download and return the HuggingFace embeddings model.

    Returns:
        HuggingFaceEmbeddings: An instance of the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings
