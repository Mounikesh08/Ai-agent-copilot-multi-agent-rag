from app.retrieval.vectordb import get_vector_store


def retrieve_relevant_chunks(query: str, k: int = 3):
    """
    Retrieve top-k relevant chunks for a user query.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    query = "What are the key points in this document?"

    results = retrieve_relevant_chunks(query, k=3)

    print(f"\nUser Query: {query}")
    print(f"Top {len(results)} relevant chunks:\n")

    for i, doc in enumerate(results, start=1):
        print("=" * 100)
        print(f"Result {i}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page_number')}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print("-" * 100)
        print(doc.page_content[:1000])
        print()