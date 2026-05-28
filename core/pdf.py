import fitz
from .parser import parse_token

def extract_from_pdf(pdf_path):
    items = []
    doc = fitz.open(pdf_path)
    for page in doc:
        for w in page.get_text("words"):
            item = parse_token(w[4].strip())
            if item:
                item["x"], item["y"] = w[0], w[1]
                items.append(item)
    return items
