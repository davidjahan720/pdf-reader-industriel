import pdfplumber

with pdfplumber.open(r'bride.pdf') as pdf:
    for page in pdf.pages:
        words = page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3)
        for w in words:
            print(f"  x={w['x0']:6.1f}  y={w['top']:6.1f}  [{w['text']}]")
