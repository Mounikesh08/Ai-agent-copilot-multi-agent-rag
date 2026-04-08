from langchain_huggingface import HuggingFaceEmbeddings
from app.utils.config import EMBEDDING_MODEL


_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    embedding_model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "embedding": vector,
            }
        )

    return embedded_chunks