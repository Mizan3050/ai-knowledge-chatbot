from pypdf import PdfReader

def extract_text_by_page(file_path: str):
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        pages.append({
            "page": i + 1,   # human-friendly (starts from 1)
            "text": text
        })

    return pages