from app.retrieval.vectordb import get_vector_store


def retrieve_relevant_chunks(query: str, k: int = 3):
    """
    Retrieve the top-k most relevant chunks from the vector store.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return results