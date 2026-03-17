from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """
    Load a PDF and return extracted text page by page.

    Returns:
        A list of dictionaries with:
        - page_number
        - text
        - source
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    pages_data = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        pages_data.append(
            {
                "page_number": i,
                "text": text.strip() if text else "",
                "source": pdf_path.name,
            }
        )

    return pages_data


if __name__ == "__main__":
    file_path = "data/raw/sample.pdf"
    pages = load_pdf(file_path)

    print(f"\nLoaded {len(pages)} pages from {file_path}\n")

    for page in pages:
        print("=" * 80)
        print(f"Source: {page['source']} | Page: {page['page_number']}")
        print("-" * 80)
        print(page["text"][:1000])  # show first 1000 characters
        print()