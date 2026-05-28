import os, ssl, tempfile, re
import cv2
import numpy as np
from PIL import Image
import fitz

ssl._create_default_https_context = ssl._create_unverified_context

import torch
GPU_AVAILABLE = torch.cuda.is_available()

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["fr", "en"], gpu=GPU_AVAILABLE, verbose=False)
    return _reader


def _pdf_to_image(pdf_path, zoom=5):
    doc  = fitz.open(pdf_path)
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    pix.save(tmp.name)
    return tmp.name


def _preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        pil = Image.open(img_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if w < 3000:
        scale = 3000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deskew
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(angle) < 10:
                angles.append(angle)
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.3:
                M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), median_angle, 1)
                gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.fastNlMeansDenoising(gray, h=8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def _run_ocr(img_path, conf_threshold=0.25):
    reader  = _get_reader()
    results = reader.readtext(
        img_path,
        rotation_info=[90, 180, 270],
        paragraph=False,
        text_threshold=0.4,
        low_text=0.25,
    )
    words = []
    for (bbox, text, conf) in results:
        if conf > conf_threshold:
            words.append({
                "text": _correct(text),
                "x": bbox[0][0],
                "y": bbox[0][1],
                "conf": conf,
            })
    return words


def extract_from_image(path, from_pdf=False):
    raw_path = None
    if from_pdf:
        raw_path = _pdf_to_image(path, zoom=5)
        img_path = raw_path
    else:
        img_path = path

    # Passe 1 : image brute (haute résolution)
    words = _run_ocr(img_path)

    # Passe 2 : image prétraitée (contraste, binarisation)
    proc = _preprocess(img_path)
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp2.close()
    cv2.imwrite(tmp2.name, proc)
    words += _run_ocr(tmp2.name)
    os.unlink(tmp2.name)

    if raw_path:
        os.unlink(raw_path)

    from .parser import parse_token
    items = []
    for w in sorted(words, key=lambda i: (i["y"]//30, i["x"])):
        item = parse_token(w["text"])
        if item:
            item["x"], item["y"] = w["x"], w["y"]
            if not _is_near_duplicate(item, items):
                items.append(item)
    return items


def _is_near_duplicate(item, existing, threshold=80):
    """Considère comme doublon : même label ET position proche (même zone du plan)."""
    lbl = item.get("label")
    x, y = item.get("x", 0), item.get("y", 0)
    for e in existing:
        if e.get("label") == lbl:
            if abs(e.get("x", 0) - x) < threshold and abs(e.get("y", 0) - y) < threshold:
                return True
    return False


def _correct(text):
    t = text.strip()
    # "1350" → "135°"
    m = re.match(r'^(\d{2,3})0$', t)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 359:
            return f"{val}°"
    # "8.2 H8" variantes OCR
    m = re.match(r'^(\d+\.?\d*)\s*[Hh]\s*[zZ]?(\d)$', t)
    if m:
        return f"{m.group(1)} H{m.group(2)}"
    # "Ra 3.2" / "Ra3,2"
    m = re.match(r'^[Rr][aAzZ]\s*(\d+\.?\d*)$', t)
    if m:
        prefix = t[:2].upper()
        return f"{prefix}{m.group(1)}"
    # "R 4.1" → "R4.1"
    m = re.match(r'^[Rr]\s+(\d+\.?\d*)$', t)
    if m:
        return f"R{m.group(1)}"
    t = re.sub(r'[_]', '.', t)
    t = t.replace(',', '.')
    return t
