#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_extract.py
Extraction de cotes depuis un plan industriel PDF via Claude Vision API.
Usage : python vision_extract.py [fichier.pdf]
Requis : ANTHROPIC_API_KEY défini en variable d'environnement
"""

import sys, os, re, glob, base64, json
import fitz          # pymupdf
import anthropic
import xlrd, xlwt
from xlutils.copy import copy as xl_copy

MODEL = "claude-haiku-4-5-20251001"

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


# ── PDF → IMAGE BASE64 ───────────────────────────────────────────────────────
def pdf_page_to_base64(pdf_path, page_index=0, dpi=200):
    doc  = fitz.open(pdf_path)
    page = doc[page_index]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    png  = pix.tobytes("png")
    return base64.standard_b64encode(png).decode("utf-8")


# ── APPEL CLAUDE VISION ──────────────────────────────────────────────────────
PROMPT = """Tu es un expert en lecture de plans industriels mécaniques.

Analyse ce plan et extrait TOUTES les cotes dans l'ordre de lecture naturel du plan (haut→bas, gauche→droite).

Retourne UNIQUEMENT un objet JSON valide, sans markdown, sans explication, avec cette structure exacte :
{
  "cotes": [
    {"type": "dimension", "valeur": 20, "label": "20"},
    {"type": "dimension", "valeur": 10, "label": "10"},
    {"type": "angle", "valeur": 135, "label": "135°"},
    {"type": "rayon", "valeur": 4.1, "label": "R4.1"},
    {"type": "iso_fit", "valeur": 40, "fit": "H7", "label": "40 H7"},
    {"type": "dimension", "valeur": 8.2, "label": "8.2"},
    {"type": "rugosite", "valeur": 3.2, "label": "Ra3.2"},
    {"type": "iso_fit", "valeur": 8.2, "fit": "H8", "label": "8.2 H8"},
    {"type": "gdt", "symbole": "⏥", "valeur": 0.1, "datum": "A", "label": "⏥ 0.1 A"},
    {"type": "gdt", "symbole": "⊕", "valeur": 0.1, "datum": "AB", "label": "⊕ 0.1 A B"},
    {"type": "gdt", "symbole": "∥", "valeur": 0.1, "datum": null, "label": "∥ 0.1"}
  ]
}

Règles :
- Inclure toutes les cotes dimensionnelles (mm), angles (°), rayons (R), rugosités (Ra), ajustements ISO (H7, H8...), tolérances GD&T
- Pour les ajustements ISO : indiquer uniquement la valeur et le code fit (ex: 40, H7) — les tolérances seront calculées
- Pour les GD&T : utiliser les symboles Unicode : ⏥ (planéité), ⊕ (position), ∥ (parallélisme), ⌖ (concentricité), ⟂ (perpendicularité)
- Ne pas inventer de cotes non visibles sur le plan
- Respecter l'ordre de lecture du plan
"""

def extract_via_vision(pdf_path):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERREUR : variable ANTHROPIC_API_KEY non définie.")
        print("  Dans le terminal : $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"  Rasterisation PDF à 200 dpi...")
    img_b64 = pdf_page_to_base64(pdf_path, dpi=200)

    print(f"  Envoi à {MODEL}...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": PROMPT}
            ],
        }]
    )

    raw = response.content[0].text.strip()

    # Nettoyage si Claude ajoute du markdown malgré les instructions
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERREUR parsing JSON : {e}")
        print("Réponse brute :", raw[:500])
        sys.exit(1)

    return data["cotes"], response.usage


# ── ENRICHISSEMENT ISO ───────────────────────────────────────────────────────
def enrich(cotes):
    for c in cotes:
        if c["type"] == "iso_fit":
            fit = c.get("fit", "")
            m = re.match(r'([A-Z])(\d+)', fit.upper())
            if m and m.group(1) == 'H':
                es, ei = h_tolerance(float(c["valeur"]), int(m.group(2)))
                if es:
                    c["es"], c["ei"] = es, ei
                    c["label"] = f"{int(c['valeur']) if float(c['valeur']) == int(c['valeur']) else c['valeur']} {fit} {es} {ei}"
    return cotes


# ── FORMATAGE AFFICHAGE ──────────────────────────────────────────────────────
def fmt(c):
    return c.get("label", "")


# ── ÉCRITURE EXCEL ────────────────────────────────────────────────────────────
def write_excel(cotes, xls_path):
    rb = xlrd.open_workbook(xls_path, formatting_info=True)
    wb = xl_copy(rb)
    ws = wb.get_sheet(0)

    row = 14  # B15 en base 0, une ligne sur deux
    for c in cotes:
        label = fmt(c)
        ws.write(row, 1, label)                  # col B : cote à relever
        if c["type"] == "gdt":
            obs = json.dumps({
                "symbole": c.get("symbole", ""),
                "valeur":  c.get("valeur"),
                "datum":   c.get("datum"),
                "cadre":   True
            }, ensure_ascii=False)
            ws.write(row, 8, obs)                # col I : OBSERVATION
        row += 2

    out = xls_path.replace(".xls", "_rempli.xls")
    wb.save(out)
    return out


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    folder = os.path.dirname(os.path.abspath(__file__))

    # PDF cible
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdfs = [f for f in glob.glob(os.path.join(folder, "*.pdf"))]
        if not pdfs:
            print("Aucun PDF trouvé.")
            sys.exit(1)
        pdf_path = pdfs[0]

    print(f"Plan : {os.path.basename(pdf_path)}")
    print(f"Modèle : {MODEL}")

    cotes, usage = extract_via_vision(pdf_path)
    cotes = enrich(cotes)

    print(f"\n── COTES EXTRAITES ({len(cotes)}) ──────────────────────────────────")
    for i, c in enumerate(cotes, 1):
        obs = ""
        if c["type"] == "gdt":
            obs = f"  →  OBSERVATION"
        print(f"  {i:2d}. {fmt(c)}{obs}")

    print(f"\n── TOKENS ──────────────────────────────────────────────────────")
    print(f"  Input: {usage.input_tokens}  |  Output: {usage.output_tokens}")

    # Excel
    xls_files = [f for f in glob.glob(os.path.join(folder, "*.xls"))
                 if "_rempli" not in f and "_out" not in f
                 and not os.path.basename(f).startswith(".~")]
    if xls_files:
        out = write_excel(cotes, xls_files[0])
        print(f"\n── EXCEL ───────────────────────────────────────────────────────")
        print(f"  {os.path.basename(out)}")
        ndim = len([c for c in cotes if c["type"] != "gdt"])
        ngdt = len([c for c in cotes if c["type"] == "gdt"])
        print(f"  {ndim} cotes + {ngdt} GD&T écrits à partir de B15")
    else:
        print("\nAucun fichier Excel trouvé.")


if __name__ == "__main__":
    main()
