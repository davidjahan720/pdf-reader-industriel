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
# Analyse complete des cadres GD&T (cellules + symbole + valeurs)
# ─────────────────────────────────────────────────────────────────────────────
_GDT_SYMBOLS_ORDER = [
    ("⏥", "planeite"),
    ("○", "circularite"),
    ("⌭", "cylindricite"),
    ("∥", "parallelisme"),
    ("⊥", "perpendicularite"),
    ("∠", "angularite"),
    ("⊕", "position"),
    ("◎", "concentricite"),
    ("≡", "symetrie"),
    ("↗", "battement"),
    ("⌒", "profil_ligne"),
    ("⌓", "profil_surface"),
]

_gdt_templates = None

def _get_gdt_templates(size=50):
    """Genere des templates Unicode pour les symboles GD&T (cache)."""
    global _gdt_templates
    if _gdt_templates is not None:
        return _gdt_templates
    from PIL import ImageDraw, ImageFont
    font = None
    for fp in ["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\seguisym.ttf",
               "arial.ttf", "DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(fp, size - 12)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    templates = {}
    for sym, _ in _GDT_SYMBOLS_ORDER:
        im = Image.new("L", (size, size), 255)
        d = ImageDraw.Draw(im)
        try:
            bb = d.textbbox((0, 0), sym, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            d.text(((size - tw) // 2 - bb[0], (size - th) // 2 - bb[1]),
                   sym, fill=0, font=font)
        except Exception:
            continue
        arr = np.array(im)
        # Skip si le symbole ne s'est pas rendu (cellule blanche)
        if np.sum(arr < 200) < 20:
            continue
        templates[sym] = arr
    _gdt_templates = templates
    return templates


def _split_frame_cells(frame_img, orientation="horizontal"):
    """Decoupe un cadre GD&T en cellules.

    orientation='horizontal' : cellules cote-a-cote (dividers verticaux)
    orientation='vertical'   : cellules empilees     (dividers horizontaux)
    Retourne liste de (start, end) le long de l'axe principal.
    """
    h, w = frame_img.shape[:2]
    if h < 8 or w < 8:
        return []
    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY) if len(frame_img.shape) == 3 else frame_img
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    if orientation == "horizontal":
        # dividers verticaux : opening avec kernel haut/fin
        k_h = max(5, int(h * 0.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
        proj = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        sums = np.sum(proj > 0, axis=0)
        threshold = h * 0.55
        axis_len = w
    else:
        # dividers horizontaux : opening avec kernel large/court
        k_w = max(5, int(w * 0.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        proj = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        sums = np.sum(proj > 0, axis=1)
        threshold = w * 0.55
        axis_len = h

    positions = np.where(sums >= threshold)[0]
    if len(positions) == 0:
        return [(0, axis_len)]

    dividers = []
    grp = [positions[0]]
    for p in positions[1:]:
        if p - grp[-1] <= 4:
            grp.append(p)
        else:
            dividers.append(int(np.mean(grp)))
            grp = [p]
    dividers.append(int(np.mean(grp)))

    margin = max(3, axis_len // 30)
    inner = [d for d in dividers if margin < d < axis_len - margin]
    if not inner:
        return [(0, axis_len)]
    cells = []
    prev = 0
    for d in inner:
        cells.append((prev, d))
        prev = d
    cells.append((prev, axis_len))
    return cells


def _match_gdt_symbol(cell_img):
    """Classifie un symbole GD&T par analyse geometrique (heuristique de forme).

    Strategie : detecter les primitives (cercles, lignes h/v/diag) puis decider :
      - cercle + croix              -> ⊕ (position)
      - 2 cercles concentriques     -> ◎ (concentricite)
      - cercle seul                 -> ○ (circularite)
      - 2+ lignes diagonales parall.-> ∥ (parallelisme)
      - vertical + horizontal       -> ⊥ (perpendicularite)
      - 3+ lignes horizontales     -> ≡ (symetrie)
      - 2 lignes horizontales      -> = (planeite simplifiee)
      - forme remplie (fill > 30%)  -> ⏥ (planeite)
    """
    from collections import Counter
    if cell_img.size == 0:
        return "?", 0.0
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
    h0, w0 = gray.shape
    if h0 < 12 or w0 < 12:
        return "?", 0.0

    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    rows = np.where(np.any(bw > 0, axis=1))[0]
    cols = np.where(np.any(bw > 0, axis=0))[0]
    if len(rows) < 5 or len(cols) < 5:
        return "?", 0.0
    content = gray[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    content_bw = bw[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    ch, cw = content.shape

    # Cercles
    circles = cv2.HoughCircles(
        content, cv2.HOUGH_GRADIENT, dp=1, minDist=max(6, ch//2),
        param1=60, param2=12,
        minRadius=max(4, min(ch, cw)//5),
        maxRadius=max(ch, cw)//2 + 2,
    )
    n_circles = circles.shape[1] if circles is not None else 0

    # Lignes Hough
    edges = cv2.Canny(content, 40, 130)
    lines_raw = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=max(6, min(ch, cw)//5),
        minLineLength=max(5, min(ch, cw)//3),
        maxLineGap=3,
    )
    horiz, vert, diag = [], [], []
    if lines_raw is not None:
        for x1, y1, x2, y2 in lines_raw[:, 0]:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
            length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            if angle < 12 or angle > 168:
                horiz.append((x1, y1, x2, y2, length, angle))
            elif 78 < angle < 102:
                vert.append((x1, y1, x2, y2, length, angle))
            else:
                diag.append((x1, y1, x2, y2, length, angle))

    fill = np.sum(content_bw > 0) / content_bw.size if content_bw.size else 0
    score = 0.75

    # ⊕ position : cercle + croix interne
    if n_circles >= 1 and (len(horiz) + len(vert)) >= 1:
        return "⊕", 0.85
    # ◎ concentricite : >= 2 cercles
    if n_circles >= 2:
        return "◎", 0.80
    # ○ circularite : 1 cercle seul
    if n_circles == 1:
        return "○", 0.75
    # ∥ parallelisme : >= 2 diagonales d'angles similaires
    if len(diag) >= 2:
        bins = Counter(int(d[5] / 12) for d in diag).most_common(1)
        if bins and bins[0][1] >= 2:
            return "∥", 0.85
    # ⊥ perpendicularite : vertical + horizontal
    if len(vert) >= 1 and len(horiz) >= 1 and not n_circles:
        return "⊥", 0.70
    # ≡ symetrie : 3+ horizontales
    if len(horiz) >= 3:
        return "≡", 0.75
    # ⏥ planeite : forme remplie ou parallelogramme
    if fill > 0.20 and not n_circles:
        return "⏥", 0.65
    # = simple lines (planeite)
    if len(horiz) >= 2 and not vert and not diag:
        return "⏥", 0.60

    return "?", 0.0


def _detect_frame_candidates(img):
    """Rectangles candidats GD&T (horizontaux ET verticaux)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    closed = cv2.dilate(bw, kh, iterations=2)
    closed = cv2.dilate(closed, kv, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                              iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0:
            continue
        # Cadre horizontal (texte normal)
        if 100 <= w <= 350 and 30 <= h <= 90 and 1.8 <= w/h <= 8.0:
            cands.append((x, y, w, h, "horizontal"))
        # Cadre vertical (texte rotated 90deg)
        if 30 <= w <= 90 and 100 <= h <= 350 and 1.8 <= h/w <= 8.0:
            cands.append((x, y, w, h, "vertical"))
    # Dedupe : garde les plus grands, supprime les chevauchements > 50%
    cands.sort(key=lambda r: -(r[2] * r[3]))
    kept = []
    for c in cands:
        cx, cy, cw, ch, _ = c
        dup = False
        for k in kept:
            kx, ky, kw, kh, _ = k
            ox = max(0, min(cx+cw, kx+kw) - max(cx, kx))
            oy = max(0, min(cy+ch, ky+kh) - max(cy, ky))
            if ox * oy > 0.5 * cw * ch:
                dup = True; break
        if not dup:
            kept.append(c)
    return kept


def _extract_gdt_items(img):
    """Pipeline complete GD&T : detection (h+v) + cellules + symbole + OCR valeurs."""
    items = []
    candidates = _detect_frame_candidates(img)
    for (x, y, w, h, orient) in candidates:
        frame = img[y:y+h, x:x+w]
        cells = _split_frame_cells(frame, orient)
        if len(cells) < 2:
            continue

        # Cellule 1 = symbole (gauche pour H, haut pour V)
        if orient == "horizontal":
            sym_cell = frame[:, cells[0][0]:cells[0][1]]
            value_crops = [frame[:, c[0]:c[1]] for c in cells[1:]]
        else:
            sym_cell = frame[cells[0][0]:cells[0][1], :]
            value_crops = [frame[c[0]:c[1], :] for c in cells[1:]]
            # Tourne les cellules de valeur de 90deg pour l'OCR
            value_crops = [cv2.rotate(vc, cv2.ROTATE_90_COUNTERCLOCKWISE) for vc in value_crops]

        symbol, sym_score = _match_gdt_symbol(sym_cell)

        # OCR sur les cellules de valeur
        text_bits = []
        for cell_img in value_crops:
            ch, cw = cell_img.shape[:2]
            # Upscale ce qui est petit
            f = max(1.5, 80 / max(ch, 1))
            cell_img = cv2.resize(cell_img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            tp = _save_temp(cell_img)
            try:
                words = _run_ocr(tp, conf_min=0.15)
            finally:
                os.unlink(tp)
            for wd in words:
                text_bits.append(wd["text"])

        # Skip si pas de valeur OCR (un vrai cadre GD&T a TOUJOURS une valeur)
        # Cela elimine les faux positifs des fleches de cotation
        if not text_bits:
            continue

        value_str = " ".join(t.replace(",", ".") for t in text_bits).strip()
        label = f"{symbol} {value_str}".strip() if value_str else symbol

        items.append({
            "type": "gdt",
            "label": label,
            "symbole": symbol if symbol != "?" else None,
            "valeur": None,
            "x": x, "y": y,
            "conf": sym_score,
            "detections": 1,
        })
    return items


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


def _merge_gdt_with_items(items, gdt_items, threshold=80):
    """Fusionne cadres GD&T detectes + heuristique sur les gdt_value orphelins."""
    def _close(a, b, t):
        return abs(a.get("x",0) - b.get("x",0)) < t and abs(a.get("y",0) - b.get("y",0)) < t

    # 1. Supprime les gdt_value couverts par un vrai cadre detecte
    out = []
    for it in items:
        if it.get("type") == "gdt_value":
            if any(_close(it, g, threshold) for g in gdt_items):
                continue
            # Sinon : applique heuristique de symbole d'apres le datum count
            datum = it.get("datum", "") or ""
            n_datums = len([c for c in datum if c.isalpha()])
            if n_datums >= 2:
                symbol = "⊕"  # 2 datums = position quasi-certain
            elif n_datums == 1:
                symbol = "⏥"  # 1 datum = planeite/parallelisme/perpendicularite (le plus frequent : planeite)
            else:
                symbol = "?"
            # Reconstruit le label proprement
            val = it.get("valeur")
            val_str = f"{val:g}" if isinstance(val, (int, float)) else ""
            datum_str = " ".join(datum) if datum else ""
            it["label"] = f"{symbol} {val_str} {datum_str}".strip()
            it["type"] = "gdt"
            it["symbole"] = symbol if symbol != "?" else None
        out.append(it)

    # 2. Ajoute les cadres GD&T detectes (dedup positionnelle stricte)
    for g in gdt_items:
        if not any(_close(g, e, 30) for e in out if e.get("type") == "gdt"):
            out.append(g)

    return out


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

    # Analyse visuelle GD&T desactivee : trop couteuse (35min) pour gain marginal.
    # On garde l'heuristique sur les gdt_value detectes par OCR (2 datums -> ⊕).
    # Les cadres GD&T sans valeur OCR-able (ex: ∥ 0.1) doivent etre ajoutes manuellement.
    gdt_items = []

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
    items = _merge_gdt_with_items(items, gdt_items)
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
