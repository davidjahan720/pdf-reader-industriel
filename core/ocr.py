"""
OCR multi-passes pour plans industriels rasterises.

Strategie :
  1. Image brute haute resolution (zoom 5x)
  2. Image pretraitee (deskew + CLAHE + threshold)
  3. Image sharpened (unsharp mask)
  4. Tiles 2x2 avec recouvrement (zoom local effectif x10)
  5. Detection des cadres GD&T -> reclassification des valeurs orphelines
  6. Filtrage du bruit (cartouche, valeurs isolees suspectes)
  7. Deduplication positionnelle (memes label & coordonnees ~ doublon)
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# Conversion / chargement
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_to_image(pdf_path, zoom=5):
    doc  = fitz.open(pdf_path)
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    pix.save(tmp.name)
    return tmp.name


def _load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        pil = Image.open(img_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _save_temp(img):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    cv2.imwrite(tmp.name, img)
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Pretraitements
# ─────────────────────────────────────────────────────────────────────────────
def _preprocess_aggressive(img):
    h, w = img.shape[:2]
    if w < 3000:
        scale = 3000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            a = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(a) < 10: angles.append(a)
        if angles:
            med = np.median(angles)
            if abs(med) > 0.3:
                M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), med, 1)
                gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.fastNlMeansDenoising(gray, h=8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    return cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)


def _sharpen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.8, blur, -0.8, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Detection des cadres GD&T (rectangles divises en cellules)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_gdt_frames(img):
    """Retourne liste [(x, y, w, h)] de cadres GD&T probables."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frames = []
    img_area = img.shape[0] * img.shape[1]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h if h else 0
        rect_area = w * h
        # Garde-fous : taille raisonnable, ratio horizontal, pas trop grand
        if (w >= 60 and h >= 18 and h <= 80
            and 2.0 < ratio < 8.0
            and rect_area < img_area * 0.02):
            # Verifie la "rectangularite" du contour
            area = cv2.contourArea(cnt)
            if rect_area > 0 and area / rect_area > 0.55:
                frames.append((x, y, w, h))
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────
def _run_ocr(img_path, conf_min=0.25):
    reader = _get_reader()
    results = reader.readtext(
        img_path,
        rotation_info=[90, 180, 270],
        paragraph=False,
        text_threshold=0.4,
        low_text=0.25,
    )
    words = []
    for (bbox, text, conf) in results:
        if conf > conf_min:
            words.append({
                "text": _correct(text),
                "x": float(bbox[0][0]),
                "y": float(bbox[0][1]),
                "conf": float(conf),
            })
    return words


def _ocr_tiles(img, tiles=2, overlap=0.20, conf_min=0.20):
    """OCR par tuiles avec recouvrement - capture le petit texte (ex. R4.1)."""
    h, w = img.shape[:2]
    tw, th = w // tiles, h // tiles
    ov_w, ov_h = int(tw * overlap), int(th * overlap)
    all_words = []
    for i in range(tiles):
        for j in range(tiles):
            x1 = max(0, i * tw - ov_w)
            y1 = max(0, j * th - ov_h)
            x2 = min(w, (i + 1) * tw + ov_w)
            y2 = min(h, (j + 1) * th + ov_h)
            tile = img[y1:y2, x1:x2]
            tp = _save_temp(tile)
            try:
                words = _run_ocr(tp, conf_min=conf_min)
            finally:
                os.unlink(tp)
            for wd in words:
                wd["x"] += x1
                wd["y"] += y1
            all_words.extend(words)
    return all_words


# ─────────────────────────────────────────────────────────────────────────────
# Filtrage / consolidation
# ─────────────────────────────────────────────────────────────────────────────
def _consolidate(parsed_items, cluster_threshold=20):
    """Clustering par position : groupe les detections proches, choisit la meilleure.

    Score (decroissant) :
      1. Nombre de detections du meme label (vote multi-passes)
      2. Confiance moyenne
      3. Longueur du label (plus de contexte = plus fiable)
      4. Penalite pour entiers courts (1-9 = plus suspects que decimales)
    """
    # Clustering single-link par position
    clusters = []
    for it in parsed_items:
        x, y = it["x"], it["y"]
        joined = False
        for cl in clusters:
            for c in cl:
                if abs(c["x"] - x) < cluster_threshold and abs(c["y"] - y) < cluster_threshold:
                    cl.append(it)
                    joined = True
                    break
            if joined:
                break
        if not joined:
            clusters.append([it])

    def _score(label, instances):
        count = len(instances)
        avg_conf = sum(i["conf"] for i in instances) / count
        llen = len(label)
        # Penalite : entiers courts (1 chiffre) sans point sont suspects
        penalty = 1.0 if (llen == 1 and label.isdigit()) else 0.0
        return (count, avg_conf, llen, -penalty)

    result = []
    for cl in clusters:
        by_label = {}
        for it in cl:
            by_label.setdefault(it["label"], []).append(it)
        ranked = sorted(by_label.items(), key=lambda kv: _score(kv[0], kv[1]), reverse=True)
        best_label, best_instances = ranked[0]
        # On garde l'instance avec la conf max pour ce label
        best = max(best_instances, key=lambda i: i["conf"])
        best["detections"] = sum(len(v) for v in by_label.values())
        result.append(best)

    return sorted(result, key=lambda i: (i.get("y", 0) // 30, i.get("x", 0)))


def _filter_noise(items, img_height):
    """Garde-fous conservateurs basés sur position, valeur et nombre de détections."""
    filtered = []
    for it in items:
        t = it.get("type")
        v = it.get("valeur")
        det = it.get("detections", 1)
        label = it.get("label", "")
        y = it.get("y", 0)

        # Garde-fou 1 : entier court (1-9) sans decimale, detecte UNE seule fois
        # = tres probablement bruit (vraie cote vue par >= 2 passes)
        if (t == "dim" and isinstance(v, (int, float)) and v <= 9
            and "." not in label and det < 2):
            continue

        # Garde-fou 2 : dans la zone cartouche, exiger conf elevee ou >= 2 detections
        if y > img_height * 0.82 and t == "dim" and isinstance(v, (int, float)):
            if v <= 9 and "." not in label and det < 3:
                continue

        # Annotation GD&T orpheline : label preserve mais marque
        if t == "gdt_value":
            if not it["label"].startswith("?"):
                it["label"] = "? " + it["label"]

        filtered.append(it)
    return filtered


def _classify_gdt(items, frames):
    """Reclassifie comme GD&T toute valeur situee dans un cadre detecte."""
    for it in items:
        x, y = it.get("x", 0), it.get("y", 0)
        for (fx, fy, fw, fh) in frames:
            if fx - 30 <= x <= fx + fw + 30 and fy - 30 <= y <= fy + fh + 30:
                it["type"] = "gdt"
                # Si label commence par "?" (parser orphelin), remplacer par symbole generique
                if it["label"].startswith("?"):
                    rest = it["label"].lstrip("? (probable GD&T)").strip()
                    it["label"] = f"⊕ {rest}"
                elif not any(s in it["label"] for s in ["⊕","∥","⏥","⌭","⊥","∠","◎","≡","↗","⌒","⌓"]):
                    it["label"] = f"⊕ {it['label']}"
                break
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entree
# ─────────────────────────────────────────────────────────────────────────────
def extract_from_image(path, from_pdf=False):
    raw_path = None
    if from_pdf:
        raw_path = _pdf_to_image(path, zoom=5)
        img_path = raw_path
    else:
        img_path = path

    img = _load_image(img_path)
    img_h = img.shape[0]
    all_words = []

    # Passe 1 : image brute haute resolution
    all_words.extend(_run_ocr(img_path, conf_min=0.25))

    # Passe 2 : image pretraitee
    proc = _preprocess_aggressive(img.copy())
    pp = _save_temp(proc)
    try:
        all_words.extend(_run_ocr(pp, conf_min=0.25))
    finally:
        os.unlink(pp)

    # Passe 3 : sharpening (souvent meilleur sur petit texte)
    sharp = _sharpen(img.copy())
    sp = _save_temp(sharp)
    try:
        all_words.extend(_run_ocr(sp, conf_min=0.20))
    finally:
        os.unlink(sp)

    # Passe 4 : OCR par tuiles 2x2 (zoom local effectif accru)
    all_words.extend(_ocr_tiles(img, tiles=2, overlap=0.20, conf_min=0.20))

    # Detection des cadres GD&T
    gdt_frames = _detect_gdt_frames(img)

    if raw_path:
        os.unlink(raw_path)

    # Parse + clustering (consolidation multi-passes)
    from .parser import parse_token
    parsed = []
    for w in all_words:
        item = parse_token(w["text"])
        if item:
            item["x"], item["y"] = w["x"], w["y"]
            item["conf"] = w.get("conf", 0)
            parsed.append(item)

    items = _consolidate(parsed, cluster_threshold=50)
    items = _classify_gdt(items, gdt_frames)
    items = _filter_noise(items, img_h)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Corrections OCR
# ─────────────────────────────────────────────────────────────────────────────
def _correct(text):
    t = text.strip()
    # "1350" -> "135°" (degre OCR comme un zero)
    m = re.match(r'^(\d{2,3})0$', t)
    if m:
        val = int(m.group(1))
        if val in (15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 270):
            return f"{val}°"
    # "8.2 H8" variantes OCR : "8.2H 8", "8.2 Hz8"
    m = re.match(r'^(\d+\.?\d*)\s*[Hh]\s*[zZ]?(\d)$', t)
    if m:
        return f"{m.group(1)} H{m.group(2)}"
    # "Ra 3.2" / "Ra3,2" / "RZ7" (OCR lit Z au lieu de z)
    m = re.match(r'^[Rr][aAzZ]\s*(\d+\.?\d*)$', t)
    if m:
        prefix = "R" + t[1].lower() if t[1].lower() in "az" else "Ra"
        return f"{prefix}{m.group(1)}"
    # "R 4.1" -> "R4.1"
    m = re.match(r'^[Rr]\s+(\d+[\.,]?\d*)$', t)
    if m:
        return f"R{m.group(1).replace(',', '.')}"
    # "R41" sans separateur -> tente "R4.1" si valeur > 10 et plausible rayon
    m = re.match(r'^[Rr](\d{2,3})$', t)
    if m and len(m.group(1)) == 2:
        # Heuristique : R41, R32, R85... probablement R4.1, R3.2, R8.5
        val = m.group(1)
        if int(val[0]) <= 9 and int(val[1]) <= 9:
            return f"R{val[0]}.{val[1]}"
    t = re.sub(r'[_]', '.', t)
    t = t.replace(',', '.')
    return t
