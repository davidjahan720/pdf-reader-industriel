import os, ssl
import cv2
import numpy as np
from PIL import Image
import fitz

ssl._create_default_https_context = ssl._create_unverified_context

# Détection GPU
import torch
GPU_AVAILABLE = torch.cuda.is_available()

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(
            ["fr", "en"],
            gpu=GPU_AVAILABLE,
            verbose=False,
        )
    return _reader


def _preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        pil = Image.open(img_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if w < 2500:
        scale = 2500 / w
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

    gray = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def extract_from_image(path, from_pdf=False):
    if from_pdf:
        doc  = fitz.open(path)
        page = doc[0]
        mat  = fitz.Matrix(3, 3)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_name = tmp.name
        tmp.close()  # fermer avant écriture (nécessaire sur Windows)
        pix.save(tmp_name)
        img_path = tmp_name
    else:
        img_path = path

    proc = _preprocess(img_path)

    import tempfile, cv2
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp2_name = tmp2.name
    tmp2.close()  # fermer avant écriture (nécessaire sur Windows)
    cv2.imwrite(tmp2_name, proc)

    reader  = _get_reader()
    results = reader.readtext(
        tmp2_name,
        rotation_info=[90, 180, 270],
        paragraph=False,
        text_threshold=0.5,
        low_text=0.3,
    )

    words = []
    for (bbox, text, conf) in results:
        if conf > 0.3:
            x = bbox[0][0]
            y = bbox[0][1]
            words.append({"text": _correct(text), "x": x, "y": y, "conf": conf})

    # Nettoyage fichiers temp
    os.unlink(tmp2_name)
    if from_pdf:
        os.unlink(img_path)

    from .parser import parse_token
    items, seen = [], set()
    for w in sorted(words, key=lambda i: (i["y"]//30, i["x"])):
        item = parse_token(w["text"])
        if item and item.get("label") not in seen:
            seen.add(item["label"])
            item["x"], item["y"] = w["x"], w["y"]
            items.append(item)
    return items


def _correct(text):
    import re
    t = text.strip()
    m = re.match(r'^(\d{2,3})0$', t)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 359:
            return f"{val}°"
    m = re.match(r'^(\d+\.?\d*)\s*[Hh]\s*[zZ]?(\d)$', t)
    if m:
        return f"{m.group(1)} H{m.group(2)}"
    t = re.sub(r'[_]', '.', t)
    t = t.replace(',', '.')
    return t
