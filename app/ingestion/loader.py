from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append(
                {
                    "source": path.name,
                    "page_number": i,
                    "text": text,
                }
            )

    return pages