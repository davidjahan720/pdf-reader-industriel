#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_engine.py — Pipeline OCR pour plans industriels
Moteur    : EasyOCR (deep learning, gère les rotations)
Fallback  : Tesseract + multi-rotation + multi-PSM
Prétraitement : upscale, deskew, CLAHE, seuillage adaptatif
"""

import os, re, ssl
import cv2
import numpy as np
from PIL import Image

# Contournement SSL pour téléchargement des modèles EasyOCR
ssl._create_default_https_context = ssl._create_unverified_context

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\David\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]

def _init_tesseract():
    import pytesseract
    for p in TESSERACT_PATHS:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return pytesseract
    return pytesseract


# ════════════════════════════════════════════════════════════════════════════
# CORRECTIONS POST-OCR — erreurs courantes sur plans techniques
# ════════════════════════════════════════════════════════════════════════════

def _correct(text):
    """Corrige les erreurs OCR typiques sur plans industriels."""
    t = text.strip()

    # ° reconnu comme 0 en fin de nombre (ex: 1350 → 135°, 450 → 45°)
    m = re.match(r'^(\d{2,3})0$', t)
    if m:
        val = int(m.group(1))
        if 10 <= val <= 359:          # plage d'angles valides
            return f"{val}°"

    # H suivi de bruit → extraire fit ISO (ex: "8H z8" → "8.2 H8")
    m = re.match(r'^(\d+\.?\d*)\s*[Hh]\s*[zZ]?(\d)$', t)
    if m:
        return f"{m.group(1)} H{m.group(2)}"

    # R collé à un chiffre mal lu (ex: "R4_1" → "R4.1")
    t = re.sub(r'^[Rr](\d+)[_,](\d+)$', r'R\1.\2', t)

    # Virgule → point dans les décimaux
    t = re.sub(r'^(\d+),(\d+)$', r'\1.\2', t)

    # O/o → 0 dans contexte numérique
    t = re.sub(r'(?<=\d)[Oo](?=\d)', '0', t)
    t = re.sub(r'(?<=\d)[Oo]$', '0', t)

    # l/I → 1 dans contexte numérique
    t = re.sub(r'^[lI](\d)', r'1\1', t)

    return t


# ════════════════════════════════════════════════════════════════════════════
# MÉTHODE A — Prétraitement avancé
# ════════════════════════════════════════════════════════════════════════════

def _deskew(gray):
    """Corrige l'inclinaison d'un scan (jusqu'à ±10°)."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:
            angles.append(angle)
    if not angles:
        return gray
    median_angle = np.median(angles)
    if abs(median_angle) < 0.3:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _upscale_if_small(img, min_width=2000):
    """Monte à min_width px si l'image est trop petite pour l'OCR."""
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return img


def preprocess_method_a(img_path_or_array):
    """
    Pipeline prétraitement avancé :
    1. Upscale si trop petit
    2. Niveaux de gris
    3. Deskew
    4. Denoise (fastNlMeans)
    5. CLAHE (contraste local adaptatif)
    6. Seuillage adaptatif (Gaussian)
    7. Dilation légère
    """
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array)
    else:
        img = img_path_or_array

    img = _upscale_if_small(img, min_width=2500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                     searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.bitwise_not(
        cv2.dilate(cv2.bitwise_not(binary), kernel, iterations=1)
    )
    return binary


# ════════════════════════════════════════════════════════════════════════════
# MÉTHODE B — Découpage par zones + multi-PSM
# ════════════════════════════════════════════════════════════════════════════

def _split_zones(img):
    """
    Découpe l'image en zones de lecture :
    - 4 quadrants (haut-gauche, haut-droite, bas-gauche, bas-droite)
    - Bande droite (souvent les cotes de côté)
    - Bande haute (souvent la cote principale)
    """
    h, w = img.shape[:2]
    zones = {
        "full":        img,
        "top":         img[0:h//3, :],
        "bottom":      img[2*h//3:, :],
        "right":       img[:, 2*w//3:],
        "left":        img[:, :w//3],
        "top_right":   img[0:h//2, w//2:],
        "mid_right":   img[h//4:3*h//4, 2*w//3:],
    }
    return zones


def _ocr_zone(zone_img, psm, pytesseract, lang="fra+eng"):
    """OCR d'une zone avec un PSM donné, retourne liste de mots."""
    data = pytesseract.image_to_data(
        zone_img, lang=lang,
        config=f"--psm {psm} --oem 3",
        output_type=pytesseract.Output.DICT
    )
    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = float(data["conf"][i])
        if text and conf > 15:
            words.append({
                "text": text,
                "conf": conf,
                "x":    data["left"][i],
                "y":    data["top"][i],
            })
    return words


def extract_method_b(preprocessed_img):
    """
    Multi-zones × multi-PSM :
    - Zones : full + 6 sous-zones
    - PSM   : 6 (bloc), 7 (ligne), 11 (sparse), 13 (raw line)
    Retourne liste de mots fusionnée et dédoublonnée.
    """
    pytesseract = _init_tesseract()
    zones = _split_zones(preprocessed_img)
    psm_modes = [6, 7, 11, 13]

    all_words = []
    for zone_name, zone_img in zones.items():
        if zone_img.size == 0:
            continue
        for psm in psm_modes:
            try:
                words = _ocr_zone(zone_img, psm, pytesseract)
                all_words.extend(words)
            except Exception:
                pass

    # Dédoublonnage par texte exact
    seen, deduped = set(), []
    for w in all_words:
        key = w["text"].lower()
        if key not in seen and len(w["text"]) >= 1:
            seen.add(key)
            deduped.append(w)

    return deduped


# ════════════════════════════════════════════════════════════════════════════
# MÉTHODE C — Multi-rotation (capture textes inclinés)
# ════════════════════════════════════════════════════════════════════════════

def _rotate_image(img, angle):
    """Rotation sans recadrage (padding blanc)."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    rotated = cv2.warpAffine(img, M, (nw, nh),
                              flags=cv2.INTER_CUBIC,
                              borderValue=255)
    return rotated


def extract_method_c(preprocessed_img):
    """
    Tourne l'image à 0°, 45°, 90°, 135°, 270°, 315° et lance l'OCR
    à chaque angle — capture les textes inclinés (cotes sur lignes de cote).
    """
    pytesseract = _init_tesseract()
    angles = [0, 45, 90, 135, 270, 315]
    all_texts = set()
    all_words = []

    for angle in angles:
        rotated = _rotate_image(preprocessed_img, angle) if angle != 0 else preprocessed_img
        words = _ocr_zone(rotated, psm=11, pytesseract=pytesseract)
        for w in words:
            key = w["text"].strip().lower()
            if key and key not in all_texts:
                all_texts.add(key)
                all_words.append(w)

    return all_words


# ════════════════════════════════════════════════════════════════════════════
# MOTEUR PRINCIPAL — EasyOCR
# ════════════════════════════════════════════════════════════════════════════

_easyocr_reader = None

def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
    return _easyocr_reader


def extract_easyocr(img_path_or_array):
    """
    EasyOCR avec rotations 0°/90°/180°/270° pour capturer les cotes inclinées.
    Retourne liste de mots avec confiance.
    """
    reader = _get_reader()

    if isinstance(img_path_or_array, str):
        img_input = img_path_or_array
    else:
        # numpy array → fichier temp
        import tempfile
        tmp = tempfile.mktemp(suffix=".png")
        cv2.imwrite(tmp, img_path_or_array)
        img_input = tmp

    results = reader.readtext(
        img_input,
        detail=1,
        rotation_info=[90, 180, 270],
        paragraph=False,
        min_size=5,
        text_threshold=0.5,
        low_text=0.3,
    )

    words = []
    for (bbox, text, conf) in results:
        text = _correct(text.strip())
        if text and conf > 0.3:
            cx = int((bbox[0][0] + bbox[2][0]) / 2)
            cy = int((bbox[0][1] + bbox[2][1]) / 2)
            words.append({"text": text, "conf": conf, "x": cx, "y": cy})

    return words


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE COMBINÉ A+B+C
# ════════════════════════════════════════════════════════════════════════════

def extract_combined(img_path):
    """
    Pipeline principal :
      1. EasyOCR (deep learning) sur image originale + prétraitée
      2. Fallback Tesseract A+B+C si EasyOCR indisponible
    Retourne liste de mots bruts dédoublonnés triés Y, X.
    """
    if isinstance(img_path, str):
        img_cv = cv2.imread(img_path)
        folder = os.path.dirname(img_path)
    else:
        img_cv = cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)
        folder = "."

    # Prétraitement A (utile pour les deux moteurs)
    preprocessed = preprocess_method_a(img_cv)
    cv2.imwrite(os.path.join(folder, "debug_preprocessed_v2.png"), preprocessed)

    try:
        # ── Moteur 1 : EasyOCR sur image originale ──────────────────────────
        words = extract_easyocr(img_path if isinstance(img_path, str) else img_cv)

        # ── Moteur 1b : EasyOCR sur image prétraitée (complément) ───────────
        words2 = extract_easyocr(preprocessed)
        seen = {w["text"].lower() for w in words}
        for w in words2:
            if w["text"].lower() not in seen:
                words.append(w)
                seen.add(w["text"].lower())

    except Exception:
        # ── Fallback : Tesseract multi-zones + multi-rotation ────────────────
        words_b = extract_method_b(preprocessed)
        words_c = extract_method_c(preprocessed)
        seen = {w["text"].lower() for w in words_b}
        for w in words_c:
            if w["text"].lower() not in seen:
                words_b.append(w)
                seen.add(w["text"].lower())
        words = words_b

    words.sort(key=lambda w: (w.get("y", 9999) // 30, w.get("x", 9999)))
    return words


# ════════════════════════════════════════════════════════════════════════════
# TEST STANDALONE
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys, glob

    folder = os.path.dirname(os.path.abspath(__file__))

    # Cherche image de test (debug ou plan)
    candidates = (glob.glob(os.path.join(folder, "debug_ocr.png")) +
                  glob.glob(os.path.join(folder, "*.png")) +
                  glob.glob(os.path.join(folder, "*.jpg")))
    if not candidates:
        print("Aucune image trouvée pour le test.")
        sys.exit(1)

    path = candidates[0]
    print(f"Test sur : {os.path.basename(path)}")
    print("Prétraitement (méthode A) + extraction multi-zones/PSM (méthode B)…\n")

    words = extract_combined(path)

    print(f"{'CONF':>4}  {'POS':>12}  TEXTE")
    print("-" * 50)
    for w in words:
        print(f"{w['conf']:4.0f}  ({w['x']:4},{w['y']:4})  {w['text']}")

    # Filtre : uniquement les tokens qui ressemblent à des cotes
    import re as _re
    COTE_RE = _re.compile(
        r'^(\d+\.?\d*[°]?$'           # nombre simple ou angle
        r'|R\d+\.?\d*$'               # rayon
        r'|Ra?\d+\.?\d*$'             # rugosité
        r'|\d+\.?\d*\s*[A-Za-z]\d+$' # fit ISO
        r'|[∥⊕⏥⌭⊥∠◎≡↗//].*$'        # GD&T
        r')'
    )
    cotes = [w for w in words if COTE_RE.match(w["text"].strip())]

    print(f"\n── RÉSUMÉ ──────────────────────────────────────────────────")
    print(f"  Mots bruts détectés : {len(words)}")
    print(f"  Cotes identifiées   : {len(cotes)}")
    print(f"\n{'CONF':>5}  COTE")
    print("-" * 30)
    for w in cotes:
        print(f"{w['conf']:5.2f}  {w['text']}")
    print("\nImage prétraitée → debug_preprocessed_v2.png")
