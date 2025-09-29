# load_data.py
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load_pdf(data_dir: str) -> List[Document]:
    """Load all PDFs from directory and return list of documents."""
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter documents to keep only 'source' metadata and page content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


def text_split(minimal_docs: List[Document]) -> List[Document]:
    """Split documents into chunks with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk
