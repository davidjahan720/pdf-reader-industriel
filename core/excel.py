import os
import xlrd, xlwt
from xlutils.copy import copy as xl_copy


def _tol_str(v):
    """Formate une valeur de tolérance : +0.025 / -0.018 / 0"""
    if v is None or v == "":
        return ""
    try:
        f = float(str(v).replace(",", "."))
        if f == 0:
            return "0"
        return f"+{f:.3f}".rstrip("0").rstrip(".") if f > 0 else f"{f:.3f}".rstrip("0").rstrip(".")
    except ValueError:
        return str(v)


def write_excel(items, xls_path):
    rb = xlrd.open_workbook(xls_path, formatting_info=True)
    wb = xl_copy(rb)
    ws = wb.get_sheet(0)
    row = 14  # première ligne de données (L14 dans le PV Meca Atlantique)

    for item in items:
        t = item.get("type", "")

        if t == "iso_fit":
            # col B (1) = cote nominale "40 H7"
            # col C (2) ligne courante  = ES (+0.025)
            # col C (2) ligne suivante  = EI (0)
            diam = item.get("valeur", "")
            zone = item.get("fit", "")
            nom  = f"{int(diam) if isinstance(diam, float) and diam == int(diam) else diam} {zone}"
            ws.write(row, 1, nom)
            es = _tol_str(item.get("es"))
            ei = _tol_str(item.get("ei"))
            if es:
                ws.write(row,     2, es)
            if ei:
                ws.write(row + 1, 2, ei)

        elif t == "gdt":
            # Symbole GD&T directement en col B, symbole à gauche de la valeur
            # Le label vient du parser avec le symbole Unicode déjà inclus (ex : "⊥ 0,05 A")
            ws.write(row, 1, item.get("label", ""))

        elif t == "rugosite":
            # Ra 3,2 — espace + virgule décimale comme dans le PV original
            valeur = item.get("valeur", "")
            prefix = item.get("label", "")[:2]  # "Ra" ou "Rz"
            val_str = str(valeur).replace(".", ",") if isinstance(valeur, float) else str(valeur)
            ws.write(row, 1, f"{prefix} {val_str}")

        else:
            # dim, rayon, angle → label tel quel en col B
            ws.write(row, 1, item.get("label", ""))

        row += 2  # cellules fusionnées : pas de 2 dans le PV

    out = xls_path.replace(".xls", "_rempli.xls")
    wb.save(out)
    return out
