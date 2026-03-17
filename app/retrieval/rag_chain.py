from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.retrieval.retriever import retrieve_relevant_chunks


MODEL_NAME = "google/flan-t5-small"


def format_context(docs) -> str:
    """
    Convert retrieved documents into a context block for the LLM.
    """
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", "unknown")

        context_parts.append(
            f"[Source {i}] File: {source}, Page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def get_local_llm():
    """
    Load a free local Hugging Face model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def generate_with_local_model(prompt: str) -> str:
    """
    Generate text using a local seq2seq model.
    """
    tokenizer, model = get_local_llm()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.0
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def generate_answer(query: str, k: int = 3) -> dict:
    """
    Retrieve relevant chunks and generate a grounded answer using a local model.
    """
    retrieved_docs = retrieve_relevant_chunks(query, k=k)
    context = format_context(retrieved_docs)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.
If the answer is not in the context, say:
"I could not find that information in the provided documents."

Be clear, accurate, and concise.
At the end, include a short Sources section listing the file names and page numbers you used.

Context:
{context}

Question:
{query}
"""

    response = generate_with_local_model(prompt)

    return {
        "question": query,
        "answer": response,
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page_number": doc.metadata.get("page_number"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
            for doc in retrieved_docs
        ],
    }


if __name__ == "__main__":
    query = "What skills are mentioned?"

    result = generate_answer(query)

    print("\n" + "=" * 100)
    print(f"Question: {result['question']}")
    print("-" * 100)
    print("Answer:")
    print(result["answer"])
    print("-" * 100)
    print("Retrieved Sources:")
    for src in result["sources"]:
        print(
            f"File: {src['source']} | Page: {src['page_number']} | Chunk ID: {src['chunk_id']}"
        )
    print("=" * 100 + "\n")