from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.ingestion.loader import load_pdf
from app.ingestion.chunking import chunk_documents
from app.ingestion.embeddings import get_embedding_model


CHROMA_DB_DIR = "data/vectordb"
COLLECTION_NAME = "documents"


def build_documents(chunks: list[dict]) -> list[Document]:
    """
    Convert chunk dictionaries into LangChain Document objects.
    """
    documents = []

    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
            },
        )
        documents.append(doc)

    return documents


def get_vector_store() -> Chroma:
    """
    Load an existing ChromaDB vector store.
    """
    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )


def store_documents(documents: list[Document]) -> Chroma:
    """
    Add documents into ChromaDB.
    """
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    return vector_store


def index_pdf(file_path: str) -> dict:
    """
    Full indexing pipeline for a PDF.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = load_pdf(file_path)
    chunks = chunk_documents(pages)
    documents = build_documents(chunks)
    store_documents(documents)

    return {
        "file_path": file_path,
        "pages_loaded": len(pages),
        "chunks_created": len(chunks),
        "documents_stored": len(documents),
    }


if __name__ == "__main__":
    file_path = "data/raw/sample.pdf"
    result = index_pdf(file_path)

    print("\nIndexing complete:")
    print(result)