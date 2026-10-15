from io import BytesIO
from pypdf import PdfReader


def extract_pdf_text(file_bytes: bytes) -> dict:
    """
    Extract text from a pdf, preseving page numbers
    """

    pdf_stream = BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    return {
        "total_pages": len(reader.pages),
        "pages": pages
    }
