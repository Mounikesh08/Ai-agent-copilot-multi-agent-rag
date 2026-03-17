from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.loader import load_pdf


def chunk_documents(pages: list[dict], chunk_size: int = 800, chunk_overlap: int = 150) -> list[dict]:
    """
    Split extracted PDF pages into smaller chunks.

    Args:
        pages: List of page dictionaries from loader.py
        chunk_size: Max size of each chunk
        chunk_overlap: Overlap between chunks for context continuity

    Returns:
        List of chunk dictionaries with text and metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []

    for page in pages:
        page_text = page["text"]

        if not page_text.strip():
            continue

        split_texts = text_splitter.split_text(page_text)

        for idx, chunk_text in enumerate(split_texts, start=1):
            all_chunks.append(
                {
                    "chunk_id": f"{page['source']}_page{page['page_number']}_chunk{idx}",
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "text": chunk_text
                }
            )

    return all_chunks


if __name__ == "__main__":
    file_path = "data/raw/sample.pdf"

    pages = load_pdf(file_path)
    chunks = chunk_documents(pages)

    print(f"\nLoaded {len(pages)} pages")
    print(f"Created {len(chunks)} chunks\n")

    for chunk in chunks[:5]:  # show first 5 chunks
        print("=" * 100)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']} | Page: {chunk['page_number']}")
        print("-" * 100)
        print(chunk["text"][:1000])
        print()