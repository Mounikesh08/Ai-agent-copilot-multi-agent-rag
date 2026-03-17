from langchain_huggingface import HuggingFaceEmbeddings
from app.ingestion.loader import load_pdf
from app.ingestion.chunking import chunk_documents


def get_embedding_model():
    """
    Load a sentence-transformer embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Convert each text chunk into an embedding vector.
    """
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


if __name__ == "__main__":
    file_path = "data/raw/sample.pdf"

    pages = load_pdf(file_path)
    chunks = chunk_documents(pages)
    embedded_chunks = embed_chunks(chunks)

    print(f"\nTotal pages loaded: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Total embedded chunks: {len(embedded_chunks)}\n")

    if embedded_chunks:
        first = embedded_chunks[0]
        print("=" * 100)
        print(f"Chunk ID: {first['chunk_id']}")
        print(f"Source: {first['source']} | Page: {first['page_number']}")
        print("-" * 100)
        print(first["text"][:500])
        print("-" * 100)
        print(f"Embedding dimension: {len(first['embedding'])}")
        print(f"First 10 values: {first['embedding'][:10]}")