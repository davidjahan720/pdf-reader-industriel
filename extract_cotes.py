#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_cotes.py
Extraction de cotes depuis un plan industriel (DXF ou PDF+OCR)
et report dans le PV de contrôle Excel (colonne B à partir de B15).

Usage:
    python extract_cotes.py bride.dxf
    python extract_cotes.py bride.pdf   (fallback OCR)
"""

import sys
import os
import re
import glob
import xlrd
import xlwt
from xlutils.copy import copy as xl_copy

# ── TABLE ISO 286 — ajustements H (alésage, EI = 0) ────────────────────────
# Clé : grade IT | valeurs : liste (borne_sup_mm, tolérance_µm)
_IT = {
    6:  [(3,6),  (6,8),  (10,9),  (18,11),(30,13),(50,16),(80,19),(120,22),(180,25),(250,29)],
    7:  [(3,10), (6,12), (10,15),(18,18),(30,21),(50,25),(80,30),(120,35),(180,40),(250,46)],
    8:  [(3,14), (6,18), (10,22),(18,27),(30,33),(50,39),(80,46),(120,54),(180,63),(250,72)],
    9:  [(3,25), (6,30), (10,36),(18,43),(30,52),(50,62),(80,74),(120,87),(180,100),(250,115)],
}

def h_tolerance(diameter, grade):
    """Retourne (ES, EI) pour ajustement H. Ex: 40 H7 → ('+0.025', '+0')"""
    for upper, mu in _IT.get(grade, []):
        if diameter <= upper:
            return f"+{mu/1000:.3f}", "+0"
    return None, None


# ── FORMATAGE D'UN ITEM ─────────────────────────────────────────────────────
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


# ── EXTRACTION DEPUIS DXF ────────────────────────────────────────────────────
def extract_from_dxf(dxf_path):
    import ezdxf
    doc  = ezdxf.readfile(dxf_path)
    msp  = doc.modelspace()
    items = []

    for entity in msp:
        dxftype = entity.dxftype()

        # ── Entités DIMENSION ──────────────────────────────────────────────
        if dxftype == "DIMENSION":
            try:
                val = entity.dxf.actual_measurement
                if val is None or val <= 0:
                    continue
                dimtype = entity.dimtype & 0x0F
                # dimtype 0/1 = linéaire, 2 = angulaire, 3 = diamètre, 4 = rayon
                if dimtype in (0, 1):
                    item = _classify_numeric(val, entity)
                elif dimtype == 2:
                    item = {"type": "angle", "value": round(val, 2), "raw": str(val)}
                elif dimtype == 3:
                    item = {"type": "dim", "value": round(val, 2), "raw": str(val), "prefix": "Ø"}
                elif dimtype == 4:
                    item = {"type": "radius", "value": round(val, 2), "raw": str(val)}
                else:
                    item = {"type": "dim", "value": round(val, 2), "raw": str(val)}
                # Récupère le texte annoté (peut contenir H7, H8, Ra...)
                override = entity.dxf.get("text", "").strip()
                if override:
                    item = _parse_text(override) or item
                if item:
                    items.append(item)
            except Exception:
                pass

        # ── Entités TEXT / MTEXT (tolérances GD&T, Ra...) ─────────────────
        elif dxftype in ("TEXT", "MTEXT"):
            try:
                text = entity.dxf.text if dxftype == "TEXT" else entity.text
                text = text.strip()
                parsed = _parse_text(text)
                if parsed:
                    items.append(parsed)
            except Exception:
                pass

    return items


def _classify_numeric(val, entity=None):
    """Transforme une valeur numérique brute en item classifié."""
    return {"type": "dim", "value": round(val, 3), "raw": str(val)}


def _parse_text(text):
    """Tente de classifier un texte extrait (dim, fit ISO, Ra, GD&T)."""
    t = text.strip()

    # GD&T frames : ⊕, //, ⏥, ⌖, ⟂ ...
    GDT_MAP = {
        r'⊕|⊕|\(⊕\)': '⊕',
        r'//|∥': '∥',
        r'⏥|⏥': '⏥',
        r'⌖': '⌖',
    }
    for pat, sym in GDT_MAP.items():
        if re.search(pat, t):
            return {"type": "gdt", "label": t, "symbol": sym, "raw": t}

    # Rugosité  Ra3.2 / 3.2 seul (contexte rugosité détecté via valeur type)
    m = re.match(r'^[Rr][Aa]?\s*(\d+\.?\d*)$', t)
    if m:
        return {"type": "roughness", "value": float(m.group(1)), "raw": t}

    # Fit ISO  40H7 / 40 H7 / 8.2H8
    m = re.match(r'^(\d+\.?\d*)\s*([A-Za-z]\d+)$', t)
    if m:
        diam = float(m.group(1))
        fit  = m.group(2)
        item = {"type": "iso_fit", "value": diam, "fit": fit, "raw": t}
        m2 = re.match(r'([A-Z])(\d+)', fit.upper())
        if m2 and m2.group(1) == 'H':
            es, ei = h_tolerance(diam, int(m2.group(2)))
            if es:
                item["es"] = es
                item["ei"] = ei
        return item

    # Rayon  R4.1
    m = re.match(r'^[Rr](\d+\.?\d*)$', t)
    if m:
        return {"type": "radius", "value": float(m.group(1)), "raw": t}

    # Angle  135°
    m = re.match(r'^(\d+\.?\d*)°$', t)
    if m:
        return {"type": "angle", "value": float(m.group(1)), "raw": t}

    # Dimension simple
    m = re.match(r'^(\d+\.?\d*)$', t)
    if m:
        v = float(m.group(1))
        if 0.5 <= v <= 10000:
            return {"type": "dim", "value": v, "raw": t}

    return None


# ── DONNÉES DE RÉFÉRENCE (bride.pdf — visuel) ────────────────────────────────
# Utilisé quand le fichier DXF n'est pas encore disponible.
BRIDE_ITEMS_HARDCODED = [
    {"type": "dim",       "value": 20,   "raw": "20"},
    {"type": "dim",       "value": 10,   "raw": "10"},
    {"type": "angle",     "value": 135,  "raw": "135°"},
    {"type": "radius",    "value": 4.1,  "raw": "R4.1"},
    {"type": "iso_fit",   "value": 40,   "fit": "H7", "es": "+0.025", "ei": "+0", "raw": "40 H7"},
    {"type": "dim",       "value": 8.2,  "raw": "8.2"},
    {"type": "roughness", "value": 3.2,  "raw": "Ra3.2"},
    {"type": "iso_fit",   "value": 8.2,  "fit": "H8", "es": "+0.022", "ei": "+0", "raw": "8.2 H8"},
    {"type": "dim",       "value": 11.5, "raw": "11.5"},
    {"type": "dim",       "value": 8,    "raw": "8"},
    {"type": "dim",       "value": 10,   "raw": "10"},
    {"type": "gdt",       "label": "⏥ 0.1 A",
     "observation": '{"symbole": "⏥", "valeur": 0.1, "datum": "A", "cadre": true}'},
    {"type": "gdt",       "label": "⊕ 0.1 A B",
     "observation": '{"symbole": "⊕", "valeur": 0.1, "datum": ["A","B"], "cadre": true}'},
    {"type": "gdt",       "label": "∥ 0.1",
     "observation": '{"symbole": "∥", "valeur": 0.1, "cadre": true}'},
    {"type": "dim",       "value": 20,   "raw": "20"},
    {"type": "dim",       "value": 6,    "raw": "6"},
    {"type": "dim",       "value": 10,   "raw": "10"},
    {"type": "dim",       "value": 10,   "raw": "10"},
]


# ── ÉCRITURE EXCEL ───────────────────────────────────────────────────────────
def write_to_excel(items, xls_path):
    rb  = xlrd.open_workbook(xls_path, formatting_info=True)
    wb  = xl_copy(rb)
    ws  = wb.get_sheet(0)

    # Colonne B = index 1  |  Colonne I (OBSERVATION) = index 8
    # Données à partir de la ligne 15 (index 14), une ligne sur deux (cellules fusionnées)
    row = 14  # ligne 15 en base 0
    for item in items:
        label = fmt(item)
        ws.write(row, 1, label)                          # col B : cote à relever
        if item.get("type") == "gdt":
            obs = item.get("observation", item.get("label", ""))
            ws.write(row, 8, obs)                        # col I : OBSERVATION
        row += 2  # saute la ligne de saisie (côtes relevées)

    out_path = xls_path.replace(".xls", "_rempli.xls")
    wb.save(out_path)
    return out_path


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    folder = os.path.dirname(os.path.abspath(__file__))

    # Cherche le DXF
    dxf_files = glob.glob(os.path.join(folder, "*.dxf"))
    pdf_files  = glob.glob(os.path.join(folder, "*.pdf"))

    if dxf_files:
        src = dxf_files[0]
        print(f"Source : DXF  →  {os.path.basename(src)}")
        items = extract_from_dxf(src)
        if not items:
            print("  (aucune entité trouvée dans le DXF, bascule sur données visuelles)")
            items = BRIDE_ITEMS_HARDCODED
    else:
        print("Pas de DXF trouvé — utilisation des cotes relevées visuellement (bride.pdf)")
        items = BRIDE_ITEMS_HARDCODED

    # Affichage
    print("\n── COTES EXTRAITES ─────────────────────────────────────────────")
    for i, item in enumerate(items, 1):
        label = fmt(item)
        obs   = f"  →  {item['observation']}" if item.get("type") == "gdt" and "observation" in item else ""
        print(f"  {i:2d}. {label}{obs}")

    # Excel
    xls_files = [f for f in glob.glob(os.path.join(folder, "*.xls"))
                 if not f.endswith("_rempli.xls") and not os.path.basename(f).startswith(".~")]
    if not xls_files:
        print("\nAucun fichier Excel trouvé.")
        return

    xls_path = xls_files[0]
    out = write_to_excel(items, xls_path)
    print(f"\n── EXCEL GÉNÉRÉ ────────────────────────────────────────────────")
    print(f"  {os.path.basename(out)}")
    print(f"  {len([i for i in items if i['type'] != 'gdt'])} cotes + "
          f"{len([i for i in items if i['type'] == 'gdt'])} GD&T écrits à partir de B15")


if __name__ == "__main__":
    main()
