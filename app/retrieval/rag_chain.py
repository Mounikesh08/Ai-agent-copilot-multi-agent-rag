from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.retrieval.retriever import retrieve_relevant_chunks
from app.utils.config import OPENAI_API_KEY, LLM_MODEL


def format_context(docs) -> str:
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", "unknown")

        context_parts.append(
            f"[Source {i}] File: {source}, Page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def get_llm():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    )


def generate_answer(query: str, k: int = 3) -> dict:
    retrieved_docs = retrieve_relevant_chunks(query, k=k)
    context = format_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI document assistant.

Answer the user's question using ONLY the provided context.
If the answer is not in the context, say:
"I could not find that information in the uploaded documents."

Keep the answer clear, concise, and user-friendly.

Context:
{context}

Question:
{question}
"""
    )

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    return {
        "question": query,
        "answer": response.content,
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page_number": doc.metadata.get("page_number"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
            for doc in retrieved_docs
        ],
    }