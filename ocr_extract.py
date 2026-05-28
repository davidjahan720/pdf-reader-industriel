#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_extract.py
Pipeline : PDF → image haute résolution → Tesseract OCR → classification cotes
"""

import sys, os, re, glob
import fitz                  # pymupdf
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

# ── CHEMIN TESSERACT (ajuste si installé ailleurs) ───────────────────────────
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\David\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]
for p in TESSERACT_PATHS:
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break


# ── TABLE ISO 286 — ajustements H (EI = 0) ──────────────────────────────────
_IT = {
    6:  [(3,6),  (6,8),  (10,9),  (18,11),(30,13),(50,16),(80,19),(120,22),(180,25),(250,29)],
    7:  [(3,10), (6,12), (10,15),(18,18),(30,21),(50,25),(80,30),(120,35),(180,40),(250,46)],
    8:  [(3,14), (6,18), (10,22),(18,27),(30,33),(50,39),(80,46),(120,54),(180,63),(250,72)],
    9:  [(3,25), (6,30), (10,36),(18,43),(30,52),(50,62),(80,74),(120,87),(180,100),(250,115)],
}

def h_tolerance(diameter, grade):
    for upper, mu in _IT.get(grade, []):
        if diameter <= upper:
            return f"+{mu/1000:.3f}", "+0"
    return None, None


# ── RASTERISATION PDF → IMAGE ────────────────────────────────────────────────
def pdf_to_image(pdf_path, page_index=0, dpi=400):
    """Convertit une page PDF en image PIL haute résolution."""
    doc  = fitz.open(pdf_path)
    page = doc[page_index]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


# ── PRÉ-TRAITEMENT IMAGE POUR OCR ────────────────────────────────────────────
def preprocess(img):
    """
    Pipeline OpenCV pour plans CAO vectorisés (traits fins) :
    1. Niveaux de gris
    2. Seuillage adaptatif (Otsu)
    3. Dilation morphologique pour épaissir les traits
    4. Retour PIL
    """
    import cv2
    import numpy as np

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Seuillage Otsu : sépare traits noirs du fond blanc
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilation 2×2 : épaissit les traits fins (caractères vectoriels 1px)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(cv2.bitwise_not(binary), kernel, iterations=1)
    result = cv2.bitwise_not(dilated)

    # Debug intermédiaire
    cv2.imwrite("debug_preprocessed.png", result)

    return Image.fromarray(result)


# ── OCR ──────────────────────────────────────────────────────────────────────
def ocr_image(img, lang="fra+eng", psm=11):
    """Lance Tesseract — PSM 11 sparse ou PSM 6 bloc selon le mode."""
    data = pytesseract.image_to_data(
        img,
        lang=lang,
        config=f"--psm {psm} --oem 3",
        output_type=pytesseract.Output.DICT
    )
    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or float(data["conf"][i]) < 15:
            continue
        words.append({
            "text": text,
            "x":    data["left"][i],
            "y":    data["top"][i],
            "conf": data["conf"][i],
        })
    return words


# ── CLASSIFICATION DES MOTS EXTRAITS ─────────────────────────────────────────
def classify(word):
    t = word["text"].strip()

    # Corrections OCR courantes sur plans techniques
    t = t.replace("°", "°").replace("O", "0").replace("l", "1") \
         .replace(",", ".").replace(";", ".")
    # ° parfois reconnu comme o ou 0 en fin de nombre
    t = re.sub(r'^(\d+\.?\d*)[oO°]$', r'\1°', t)

    # Rugosité : Ra3.2 / Ra 3.2 / 3.2 (contexte rugosité à affiner plus tard)
    m = re.match(r'^[Rr][Aa]?\s*(\d+\.?\d*)$', t)
    if m:
        return {"type": "roughness", "value": float(m.group(1)), "raw": t}

    # Fit ISO : 40H7 / 40 H7 / 8.2H8
    m = re.match(r'^(\d+\.?\d*)\s*([A-Za-z]\d+)$', t)
    if m:
        diam = float(m.group(1))
        fit  = m.group(2).upper()
        item = {"type": "iso_fit", "value": diam, "fit": fit, "raw": t}
        m2 = re.match(r'([A-Z])(\d+)', fit)
        if m2 and m2.group(1) == 'H':
            es, ei = h_tolerance(diam, int(m2.group(2)))
            if es:
                item["es"], item["ei"] = es, ei
        return item

    # Rayon : R4.1
    m = re.match(r'^[Rr](\d+\.?\d*)$', t)
    if m:
        return {"type": "radius", "value": float(m.group(1)), "raw": t}

    # Angle : 135°
    m = re.match(r'^(\d+\.?\d*)°$', t)
    if m:
        return {"type": "angle", "value": float(m.group(1)), "raw": t}

    # Dimension simple : 8.2 / 11.5 / 40
    m = re.match(r'^(\d+\.?\d*)$', t)
    if m:
        v = float(m.group(1))
        if 0.5 <= v <= 9999:
            return {"type": "dim", "value": v, "raw": t}

    return None


# ── DÉDOUBLONNAGE ─────────────────────────────────────────────────────────────
def deduplicate(items, tol=0.01):
    """Supprime les doublons exacts (même type + valeur arrondie)."""
    seen = set()
    result = []
    for item in items:
        key = (item["type"], round(item.get("value", 0), 2), item.get("fit", ""))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ── FORMATAGE ─────────────────────────────────────────────────────────────────
def fmt(item):
    t = item["type"]
    if t == "dim":
        v = item["value"]
        return str(int(v)) if v == int(v) else str(v)
    if t == "radius":
        v = item["value"]
        return f"R{int(v) if v == int(v) else v}"
    if t == "angle":
        v = item["value"]
        return f"{int(v) if v == int(v) else v}°"
    if t == "roughness":
        v = item["value"]
        return f"Ra{int(v) if v == int(v) else v}"
    if t == "iso_fit":
        diam = item["value"]
        fit  = item["fit"]
        base = f"{int(diam) if diam == int(diam) else diam} {fit}"
        if "es" in item:
            return f"{base} {item['es']} {item['ei']}"
        return base
    if t == "gdt":
        return item["label"]
    return item.get("raw", "")


# ── GD&T (toujours hardcodé : symboles = vecteurs) ───────────────────────────
GDT_ITEMS = [
    {"type": "gdt", "label": "⏥ 0.1 A",
     "observation": '{"symbole": "⏥", "valeur": 0.1, "datum": "A", "cadre": true}'},
    {"type": "gdt", "label": "⊕ 0.1 A B",
     "observation": '{"symbole": "⊕", "valeur": 0.1, "datum": ["A","B"], "cadre": true}'},
    {"type": "gdt", "label": "∥ 0.1",
     "observation": '{"symbole": "∥", "valeur": 0.1, "cadre": true}'},
]


# ── ÉCRITURE EXCEL ────────────────────────────────────────────────────────────
def write_excel(items, xls_path):
    import xlrd, xlwt
    from xlutils.copy import copy as xl_copy

    rb = xlrd.open_workbook(xls_path, formatting_info=True)
    wb = xl_copy(rb)
    ws = wb.get_sheet(0)

    row = 14  # B15 en base 0
    for item in items:
        label = fmt(item)
        ws.write(row, 1, label)
        if item.get("type") == "gdt":
            ws.write(row, 8, item.get("observation", ""))
        row += 2

    out = xls_path.replace(".xls", "_rempli.xls")
    if out == xls_path:
        out = xls_path.replace(".xls", "_out.xls")
    wb.save(out)
    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    pdfs   = glob.glob(os.path.join(folder, "*.pdf"))

    if not pdfs:
        print("Aucun PDF trouvé dans", folder)
        sys.exit(1)

    pdf_path = pdfs[0]
    print(f"PDF : {os.path.basename(pdf_path)}")

    # Test Tesseract
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"Tesseract : v{ver}")
    except Exception as e:
        print(f"ERREUR Tesseract non trouvé : {e}")
        print("Installe Tesseract OCR depuis :")
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)

    print("Rasterisation du PDF à 600 dpi...")
    img = pdf_to_image(pdf_path, dpi=600)
    img_pre = preprocess(img)

    # Sauvegarde de l'image pour debug
    debug_path = os.path.join(folder, "debug_ocr.png")
    img_pre.save(debug_path)
    print(f"Image debug : {os.path.basename(debug_path)}")

    print("OCR en cours (passe 1 — sparse)...")
    words = ocr_image(img_pre)
    print(f"  {len(words)} mots détectés")

    # Passe 2 : PSM 6 (bloc uniforme) pour attraper les cotes isolées
    print("OCR en cours (passe 2 — bloc)...")
    words2 = ocr_image(img_pre, lang="fra+eng", psm=6)
    # Fusion sans doublons (même texte + position proche)
    existing = {(w["text"], w["x"]//30, w["y"]//30) for w in words}
    for w in words2:
        key = (w["text"], w["x"]//30, w["y"]//30)
        if key not in existing:
            words.append(w)
            existing.add(key)
    print(f"  {len(words)} mots au total après fusion")

    # Affichage brut pour diagnostic
    print("\n── MOTS BRUTS OCR (confiance > 20) ────────────────────────────")
    for w in words:
        print(f"  conf={w['conf']:3}  ({w['x']:4},{w['y']:4})  [{w['text']}]")

    # Classification
    items_ocr = []
    for w in words:
        item = classify(w)
        if item:
            item["x"] = w["x"]
            item["y"] = w["y"]
            items_ocr.append(item)

    # Tri par position Y (haut → bas) puis X (gauche → droite)
    items_ocr.sort(key=lambda i: (i.get("y", 0) // 20, i.get("x", 0)))
    items_ocr = deduplicate(items_ocr)

    # Injection GD&T (toujours hardcodé)
    # On les insère après le 11e item (position connue dans ce plan)
    INSERT_POS = 11
    items_final = items_ocr[:INSERT_POS] + GDT_ITEMS + items_ocr[INSERT_POS:]

    print("\n── COTES CLASSIFIÉES ───────────────────────────────────────────")
    for i, item in enumerate(items_final, 1):
        label = fmt(item)
        obs = f"  →  {item['observation']}" if item.get("type") == "gdt" else ""
        print(f"  {i:2d}. {label}{obs}")

    # Excel
    xls_files = [f for f in glob.glob(os.path.join(folder, "*.xls"))
                 if "_rempli" not in f and "_out" not in f
                 and not os.path.basename(f).startswith(".~")]
    if xls_files:
        out = write_excel(items_final, xls_files[0])
        print(f"\n── EXCEL ───────────────────────────────────────────────────────")
        print(f"  Généré : {os.path.basename(out)}")
        print(f"  {len([i for i in items_final if i['type'] != 'gdt'])} cotes + "
              f"{len([i for i in items_final if i['type'] == 'gdt'])} GD&T")
    else:
        print("\nAucun fichier Excel trouvé.")


if __name__ == "__main__":
    main()
