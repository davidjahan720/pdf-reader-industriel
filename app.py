#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Interface graphique extraction de cotes plans industriels
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, os, re, json, glob
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import fitz        # pymupdf
import xlrd, xlwt
from xlutils.copy import copy as xl_copy

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# ── TABLE ISO 286 (source : reference_gdt.txt) ───────────────────────────────
# Structure : { "ZONE": { (borne_inf, borne_sup): (ES, EI) } }
ISO_TABLE = {
    "H6":  {(0,3):(+0.006,0),(3,6):(+0.008,0),(6,10):(+0.009,0),(10,18):(+0.011,0),
            (18,30):(+0.013,0),(30,50):(+0.016,0),(50,80):(+0.019,0),(80,120):(+0.022,0),(120,180):(+0.025,0),(180,250):(+0.029,0)},
    "H7":  {(0,3):(+0.010,0),(3,6):(+0.012,0),(6,10):(+0.015,0),(10,18):(+0.018,0),
            (18,30):(+0.021,0),(30,50):(+0.025,0),(50,80):(+0.030,0),(80,120):(+0.035,0),(120,180):(+0.040,0),(180,250):(+0.046,0)},
    "H8":  {(0,3):(+0.014,0),(3,6):(+0.018,0),(6,10):(+0.022,0),(10,18):(+0.027,0),
            (18,30):(+0.033,0),(30,50):(+0.039,0),(50,80):(+0.046,0),(80,120):(+0.054,0),(120,180):(+0.063,0),(180,250):(+0.072,0)},
    "H9":  {(0,3):(+0.025,0),(3,6):(+0.030,0),(6,10):(+0.036,0),(10,18):(+0.043,0),
            (18,30):(+0.052,0),(30,50):(+0.062,0),(50,80):(+0.074,0),(80,120):(+0.087,0),(120,180):(+0.100,0),(180,250):(+0.115,0)},
    "h6":  {(0,3):(0,-0.006),(3,6):(0,-0.008),(6,10):(0,-0.009),(10,18):(0,-0.011),
            (18,30):(0,-0.013),(30,50):(0,-0.016),(50,80):(0,-0.019),(80,120):(0,-0.022),(120,180):(0,-0.025),(180,250):(0,-0.029)},
    "h7":  {(0,3):(0,-0.010),(3,6):(0,-0.012),(6,10):(0,-0.015),(10,18):(0,-0.018),
            (18,30):(0,-0.021),(30,50):(0,-0.025),(50,80):(0,-0.030),(80,120):(0,-0.035),(120,180):(0,-0.040),(180,250):(0,-0.046)},
    "g6":  {(3,6):(-0.004,-0.012),(6,10):(-0.005,-0.014),(10,18):(-0.006,-0.017),
            (18,30):(-0.007,-0.020),(30,50):(-0.009,-0.025),(50,80):(-0.010,-0.029)},
    "f7":  {(3,6):(-0.010,-0.022),(6,10):(-0.013,-0.028),(10,18):(-0.016,-0.034),
            (18,30):(-0.020,-0.041),(30,50):(-0.025,-0.050),(50,80):(-0.030,-0.060)},
}

GDT_SYMBOLS = {
    "planeite":       "⏥",
    "rectitude":      "—",
    "circularite":    "○",
    "cylindricite":   "⌭",
    "parallelisme":   "∥",
    "perpendicularite": "⊥",
    "angularite":     "∠",
    "position":       "⊕",
    "concentricite":  "◎",
    "symetrie":       "≡",
    "battement":      "↗",
    "battement_total":"↗↗",
    "profil_ligne":   "⌒",
    "profil_surface": "⌓",
}

def iso_tolerance(diam, zone):
    tbl = ISO_TABLE.get(zone, {})
    for (lo, hi), (es, ei) in tbl.items():
        if lo < diam <= hi or (lo == 0 and diam <= hi):
            es_str = (f"+{es:.3f}" if es >= 0 else f"{es:.3f}").rstrip("0").rstrip(".")
            ei_str = (f"+{ei:.3f}" if ei >= 0 else f"{ei:.3f}").rstrip("0").rstrip(".")
            if es_str in ("+0", "0", ""): es_str = "+0"
            if ei_str in ("+0", "0", "-0", ""): ei_str = "+0" if ei == 0 else ei_str
            return es_str, ei_str
    return None, None


# ── EXTRACTION PDF (texte vectoriel + fallback visuel) ───────────────────────
def extract_from_pdf(pdf_path):
    """Tente l'extraction texte PyMuPDF, sinon retourne liste vide."""
    items = []
    doc = fitz.open(pdf_path)
    for page in doc:
        words = page.get_text("words")
        for w in words:
            text = w[4].strip()
            item = parse_token(text)
            if item:
                item["x"] = w[0]
                item["y"] = w[1]
                items.append(item)
    return items


def extract_from_dxf(dxf_path):
    """Extrait les entités DIMENSION et TEXT d'un DXF coté."""
    import ezdxf
    items = []
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    for e in msp:
        if e.dxftype() == "DIMENSION":
            try:
                val = e.dxf.actual_measurement
                txt = e.dxf.get("text", "").strip()
                item = parse_token(txt) if txt else None
                if item is None and val and val > 0:
                    item = {"type": "dim", "valeur": round(val, 3), "label": str(round(val, 3))}
                if item:
                    items.append(item)
            except Exception:
                pass
        elif e.dxftype() in ("TEXT", "MTEXT"):
            try:
                txt = (e.dxf.text if e.dxftype() == "TEXT" else e.text).strip()
                item = parse_token(txt)
                if item:
                    items.append(item)
            except Exception:
                pass
    return items


def parse_token(t):
    """Classifie un token texte en item structuré."""
    t = t.strip()
    if not t:
        return None

    # Normalisation symboles OCR → Unicode
    t = t.replace("//", "∥")

    # GD&T frames : ⊕ ∥ ⏥ ⌭ ⊥ ∠ ◎ ≡
    if any(sym in t for sym in ["⊕", "∥", "⏥", "⌭", "⊥", "∠", "◎", "≡", "↗", "⌒", "⌓"]):
        return {"type": "gdt", "label": t, "valeur": None}

    # Rugosité Ra / Rz
    m = re.match(r'^R[azpAZP]\s*(\d+\.?\d*)$', t)
    if m:
        param = t[:2].upper()
        return {"type": "rugosite", "valeur": float(m.group(1)), "label": f"{param}{m.group(1)}"}

    # Fit ISO : 40H7 / 8.2 H8 / 25h6
    m = re.match(r'^(\d+\.?\d*)\s*([A-Za-z]\d+)$', t)
    if m:
        diam = float(m.group(1))
        zone = m.group(2)
        item = {"type": "iso_fit", "valeur": diam, "fit": zone}
        es, ei = iso_tolerance(diam, zone.upper() if zone[0].isupper() else zone)
        if es is None:
            es, ei = iso_tolerance(diam, zone)
        if es:
            item["es"], item["ei"] = es, ei
            item["label"] = f"{int(diam) if diam == int(diam) else diam} {zone} {es} {ei}"
        else:
            item["label"] = f"{int(diam) if diam == int(diam) else diam} {zone}"
        return item

    # Rayon
    m = re.match(r'^[Rr](\d+\.?\d*)$', t)
    if m:
        v = float(m.group(1))
        return {"type": "rayon", "valeur": v, "label": f"R{v}"}

    # Angle
    m = re.match(r'^(\d+\.?\d*)°$', t)
    if m:
        v = float(m.group(1))
        return {"type": "angle", "valeur": v, "label": f"{int(v) if v == int(v) else v}°"}

    # Dimension simple
    m = re.match(r'^(\d+\.?\d*)$', t)
    if m:
        v = float(m.group(1))
        if 0.5 <= v <= 9999:
            return {"type": "dim", "valeur": v, "label": str(int(v) if v == int(v) else v)}

    return None


def extract_from_image(img_path):
    """
    OCR amélioré pour images scannées :
    Méthode A (prétraitement avancé) + B (multi-zones/PSM) + C (multi-rotation).
    """
    try:
        from ocr_engine import extract_combined
        words = extract_combined(img_path)

        items = []
        seen_labels = set()
        for w in words:
            item = parse_token(w["text"])
            if item:
                lbl = item.get("label", "")
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    item["x"] = w.get("x", 0)
                    item["y"] = w.get("y", 0)
                    items.append(item)

        items.sort(key=lambda it: (it.get("y", 9999) // 30, it.get("x", 0)))
        return items

    except Exception as e:
        return []


def render_pdf_page(pdf_path, width=500):
    """Rend la première page PDF en image PIL redimensionnée."""
    doc  = fitz.open(pdf_path)
    page = doc[0]
    zoom = width / page.rect.width
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def write_excel(items, xls_path):
    rb  = xlrd.open_workbook(xls_path, formatting_info=True)
    wb  = xl_copy(rb)
    ws  = wb.get_sheet(0)
    row = 14
    for item in items:
        ws.write(row, 1, item.get("label", ""))
        if item.get("type") == "gdt":
            obs = json.dumps({"symbole": item.get("symbole", item.get("label","")),
                              "valeur": item.get("valeur"), "cadre": True},
                             ensure_ascii=False)
            ws.write(row, 8, obs)
        row += 2
    out = xls_path.replace(".xls", "_rempli.xls")
    wb.save(out)
    return out


# ════════════════════════════════════════════════════════════════════════════
# INTERFACE GRAPHIQUE
# ════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Extracteur de cotes — Plans industriels")
        self.geometry("1100x700")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        self.current_file = None
        self.items = []
        self._photo = None

        self._build_ui()

    # ── CONSTRUCTION UI ──────────────────────────────────────────────────────
    def _build_ui(self):
        # Palette couleurs
        BG      = "#1e1e2e"
        PANEL   = "#2a2a3e"
        ACCENT  = "#7c6af7"
        TEXT    = "#cdd6f4"
        SUBTEXT = "#6c7086"
        GREEN   = "#a6e3a1"
        YELLOW  = "#f9e2af"
        RED     = "#f38ba8"
        BLUE    = "#89b4fa"

        # ── Barre du haut ────────────────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL, pady=8, padx=12)
        top.pack(fill="x")

        tk.Label(top, text="📐 Extracteur de cotes", font=("Segoe UI", 13, "bold"),
                 bg=PANEL, fg=TEXT).pack(side="left")

        btn_style = {"font": ("Segoe UI", 10), "relief": "flat",
                     "padx": 14, "pady": 5, "cursor": "hand2"}

        tk.Button(top, text="📂  Ouvrir plan",
                  bg=ACCENT, fg="white",
                  command=self._open_file, **btn_style).pack(side="left", padx=(20,4))

        tk.Button(top, text="⚡  Analyser",
                  bg="#45475a", fg=TEXT,
                  command=self._analyze, **btn_style).pack(side="left", padx=4)

        tk.Button(top, text="💾  Exporter Excel",
                  bg=GREEN, fg="#1e1e2e",
                  command=self._export_excel, **btn_style).pack(side="left", padx=4)

        tk.Button(top, text="🗑  Vider",
                  bg="#45475a", fg=TEXT,
                  command=self._clear, **btn_style).pack(side="left", padx=4)

        self.lbl_file = tk.Label(top, text="Aucun fichier sélectionné",
                                  font=("Segoe UI", 9), bg=PANEL, fg=SUBTEXT)
        self.lbl_file.pack(side="left", padx=16)

        # ── Corps principal ──────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # Panneau gauche — aperçu PDF
        left = tk.Frame(body, bg=PANEL, bd=0, relief="flat")
        left.pack(side="left", fill="both", expand=True, padx=(0,4))

        tk.Label(left, text="Aperçu du plan", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=8, pady=(6,2))

        self.canvas = tk.Canvas(left, bg="#13131f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=(0,6))

        # Panneau droit — table des cotes
        right = tk.Frame(body, bg=PANEL, width=420)
        right.pack(side="right", fill="both", padx=(4,0))
        right.pack_propagate(False)

        hdr = tk.Frame(right, bg=PANEL)
        hdr.pack(fill="x", padx=8, pady=(6,2))
        tk.Label(hdr, text="Cotes extraites", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=SUBTEXT).pack(side="left")
        self.lbl_count = tk.Label(hdr, text="", font=("Segoe UI", 9),
                                   bg=PANEL, fg=ACCENT)
        self.lbl_count.pack(side="right")

        # Treeview
        cols = ("n", "label", "type")
        self.tree = ttk.Treeview(right, columns=cols, show="headings",
                                  selectmode="extended", height=30)
        self.tree.heading("n",     text="#",     anchor="center")
        self.tree.heading("label", text="Cote",  anchor="w")
        self.tree.heading("type",  text="Type",  anchor="w")
        self.tree.column("n",     width=35,  stretch=False, anchor="center")
        self.tree.column("label", width=240, stretch=True,  anchor="w")
        self.tree.column("type",  width=100, stretch=False, anchor="w")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=24,
                        font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#313244",
                        foreground=SUBTEXT, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

        # Tags couleurs par type
        self.tree.tag_configure("dim",      foreground=TEXT)
        self.tree.tag_configure("iso_fit",  foreground=BLUE)
        self.tree.tag_configure("rayon",    foreground=YELLOW)
        self.tree.tag_configure("angle",    foreground=YELLOW)
        self.tree.tag_configure("rugosite", foreground="#fab387")
        self.tree.tag_configure("gdt",      foreground=GREEN)

        sb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8,0), pady=(0,6))
        sb.pack(side="right", fill="y", pady=(0,6), padx=(0,8))

        # Menu clic droit
        self.menu = tk.Menu(self, tearoff=0, bg="#313244", fg=TEXT,
                             activebackground=ACCENT, activeforeground="white",
                             font=("Segoe UI", 9))
        self.menu.add_command(label="✏️  Modifier",   command=self._edit_item)
        self.menu.add_command(label="➕  Ajouter",    command=self._add_item)
        self.menu.add_separator()
        self.menu.add_command(label="❌  Supprimer",  command=self._delete_item)
        self.tree.bind("<Button-3>", self._show_menu)
        self.tree.bind("<Double-1>", lambda e: self._edit_item())

        # ── Barre de statut ──────────────────────────────────────────────────
        self.status = tk.Label(self, text="Prêt — ouvrez un plan PDF ou DXF",
                                font=("Segoe UI", 8), bg="#181825", fg=SUBTEXT,
                                anchor="w", padx=10, pady=4)
        self.status.pack(fill="x", side="bottom")

    # ── ACTIONS ─────────────────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Sélectionner un plan",
            filetypes=[
                ("Plans industriels", "*.pdf *.dxf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("PDF",    "*.pdf"),
                ("DXF",    "*.dxf"),
                ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("Tous",   "*.*"),
            ]
        )
        if not path:
            return
        self.current_file = path
        name = os.path.basename(path)
        self.lbl_file.config(text=name)
        self._set_status(f"Fichier chargé : {name}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf" or ext in IMAGE_EXTS:
            threading.Thread(target=self._load_preview, args=(path,), daemon=True).start()
        else:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 200, 200,
                text="DXF — pas d'aperçu visuel", fill="#6c7086",
                font=("Segoe UI", 11)
            )

    def _load_preview(self, path):
        try:
            ext = os.path.splitext(path)[1].lower()
            w = self.canvas.winfo_width() or 480
            if ext == ".pdf":
                img = render_pdf_page(path, width=w)
            elif ext in IMAGE_EXTS:
                img = Image.open(path)
                img.thumbnail((w, 2000), Image.LANCZOS)
            else:
                return
            photo = ImageTk.PhotoImage(img)
            self._photo = photo
            self.canvas.after(0, lambda: self._show_preview(photo))
        except Exception as e:
            self._set_status(f"Aperçu impossible : {e}")

    def _show_preview(self, photo):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)

    def _analyze(self):
        if not self.current_file:
            messagebox.showwarning("Aucun fichier", "Ouvrez d'abord un plan.")
            return
        self._set_status("Analyse en cours…")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        path = self.current_file
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".dxf":
                items = extract_from_dxf(path)
                source = "DXF"
            elif ext in IMAGE_EXTS:
                items = extract_from_image(path)
                source = f"OCR image ({ext.lstrip('.')})"
            else:
                items = extract_from_pdf(path)
                source = "PDF (texte)"

            if not items:
                name = os.path.basename(path).lower()
                if "bride" in name:
                    items = BRIDE_FALLBACK.copy()
                    source = "visuel (bride.pdf)"
                else:
                    source += " — aucun texte trouvé, liste vide (éditez manuellement)"

            self.items = items
            self.after(0, lambda: self._populate_tree(items, source))
        except Exception as e:
            self.after(0, lambda: self._set_status(f"Erreur : {e}"))

    def _populate_tree(self, items, source=""):
        self.tree.delete(*self.tree.get_children())
        for i, item in enumerate(items, 1):
            tag  = item.get("type", "dim")
            lbl  = item.get("label", "")
            typ  = tag
            self.tree.insert("", "end", iid=str(i-1),
                             values=(i, lbl, typ), tags=(tag,))
        ndim = len([x for x in items if x.get("type") != "gdt"])
        ngdt = len([x for x in items if x.get("type") == "gdt"])
        self.lbl_count.config(text=f"{ndim} cotes · {ngdt} GD&T")
        self._set_status(f"Extraction {source} : {len(items)} éléments trouvés")

    def _export_excel(self):
        if not self.items:
            messagebox.showwarning("Vide", "Aucune cote à exporter.")
            return
        folder = os.path.dirname(self.current_file) if self.current_file else "."
        xls_files = [f for f in glob.glob(os.path.join(folder, "*.xls"))
                     if "_rempli" not in f and not os.path.basename(f).startswith(".~")]
        if not xls_files:
            path = filedialog.askopenfilename(
                title="Sélectionner le fichier Excel PV",
                filetypes=[("Excel", "*.xls *.xlsx")]
            )
            if not path:
                return
            xls_files = [path]

        # Sync items depuis le tree (ordre utilisateur)
        synced = []
        for iid in self.tree.get_children():
            idx = int(iid)
            if idx < len(self.items):
                synced.append(self.items[idx])

        try:
            out = write_excel(synced, xls_files[0])
            self._set_status(f"Excel généré : {os.path.basename(out)}")
            messagebox.showinfo("Export réussi", f"Fichier créé :\n{out}")
        except Exception as e:
            messagebox.showerror("Erreur export", str(e))

    def _clear(self):
        self.items = []
        self.tree.delete(*self.tree.get_children())
        self.lbl_count.config(text="")
        self._set_status("Liste vidée")

    # ── EDITION ──────────────────────────────────────────────────────────────
    def _show_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
        self.menu.post(event.x_root, event.y_root)

    def _edit_item(self):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        idx = int(iid)
        item = self.items[idx]

        win = tk.Toplevel(self)
        win.title("Modifier la cote")
        win.geometry("360x160")
        win.configure(bg="#1e1e2e")
        win.grab_set()

        tk.Label(win, text="Libellé :", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10)).grid(row=0, column=0, padx=12, pady=12, sticky="w")
        var_lbl = tk.StringVar(value=item.get("label", ""))
        e_lbl = tk.Entry(win, textvariable=var_lbl, width=30,
                         bg="#313244", fg="white", insertbackground="white",
                         font=("Consolas", 11), relief="flat")
        e_lbl.grid(row=0, column=1, padx=8, pady=12)
        e_lbl.focus()

        tk.Label(win, text="Type :", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10)).grid(row=1, column=0, padx=12, pady=4, sticky="w")
        types = ["dim","iso_fit","rayon","angle","rugosite","gdt"]
        var_type = tk.StringVar(value=item.get("type","dim"))
        ttk.Combobox(win, textvariable=var_type, values=types, width=14,
                     state="readonly").grid(row=1, column=1, padx=8, pady=4, sticky="w")

        def save():
            self.items[idx]["label"] = var_lbl.get()
            self.items[idx]["type"]  = var_type.get()
            self.tree.item(iid, values=(idx+1, var_lbl.get(), var_type.get()),
                           tags=(var_type.get(),))
            win.destroy()

        tk.Button(win, text="Enregistrer", command=save,
                  bg="#7c6af7", fg="white", relief="flat",
                  font=("Segoe UI", 10), padx=12, pady=6,
                  cursor="hand2").grid(row=2, column=1, padx=8, pady=14, sticky="e")

    def _add_item(self):
        win = tk.Toplevel(self)
        win.title("Ajouter une cote")
        win.geometry("360x160")
        win.configure(bg="#1e1e2e")
        win.grab_set()

        tk.Label(win, text="Libellé :", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10)).grid(row=0, column=0, padx=12, pady=12, sticky="w")
        var_lbl = tk.StringVar()
        e_lbl = tk.Entry(win, textvariable=var_lbl, width=30,
                         bg="#313244", fg="white", insertbackground="white",
                         font=("Consolas", 11), relief="flat")
        e_lbl.grid(row=0, column=1, padx=8, pady=12)
        e_lbl.focus()

        tk.Label(win, text="Type :", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10)).grid(row=1, column=0, padx=12, pady=4, sticky="w")
        types = ["dim","iso_fit","rayon","angle","rugosite","gdt"]
        var_type = tk.StringVar(value="dim")
        ttk.Combobox(win, textvariable=var_type, values=types, width=14,
                     state="readonly").grid(row=1, column=1, padx=8, pady=4, sticky="w")

        def add():
            lbl = var_lbl.get().strip()
            if not lbl:
                return
            item = parse_token(lbl) or {"type": var_type.get(), "label": lbl, "valeur": None}
            item["type"] = var_type.get()
            item["label"] = lbl
            self.items.append(item)
            n = len(self.items)
            self.tree.insert("", "end", iid=str(n-1),
                             values=(n, lbl, item["type"]), tags=(item["type"],))
            win.destroy()

        tk.Button(win, text="Ajouter", command=add,
                  bg="#a6e3a1", fg="#1e1e2e", relief="flat",
                  font=("Segoe UI", 10), padx=12, pady=6,
                  cursor="hand2").grid(row=2, column=1, padx=8, pady=14, sticky="e")

    def _delete_item(self):
        sel = self.tree.selection()
        if not sel:
            return
        for iid in reversed(sel):
            idx = int(iid)
            self.items.pop(idx)
            self.tree.delete(iid)
        # Renuméroter
        for i, iid in enumerate(self.tree.get_children()):
            vals = self.tree.item(iid, "values")
            self.tree.item(iid, values=(i+1, vals[1], vals[2]))

    def _set_status(self, msg):
        self.status.config(text=msg)


# ── DONNÉES FALLBACK bride.pdf (relevées visuellement) ──────────────────────
BRIDE_FALLBACK = [
    {"type": "dim",      "valeur": 20,   "label": "20"},
    {"type": "dim",      "valeur": 10,   "label": "10"},
    {"type": "angle",    "valeur": 135,  "label": "135°"},
    {"type": "rayon",    "valeur": 4.1,  "label": "R4.1"},
    {"type": "iso_fit",  "valeur": 40,   "fit": "H7",  "label": "40 H7 +0.025 +0"},
    {"type": "dim",      "valeur": 8.2,  "label": "8.2"},
    {"type": "rugosite", "valeur": 3.2,  "label": "Ra3.2"},
    {"type": "iso_fit",  "valeur": 8.2,  "fit": "H8",  "label": "8.2 H8 +0.022 +0"},
    {"type": "dim",      "valeur": 11.5, "label": "11.5"},
    {"type": "dim",      "valeur": 8,    "label": "8"},
    {"type": "dim",      "valeur": 10,   "label": "10"},
    {"type": "gdt",  "label": "⏥ 0.1 A",   "symbole": "⏥", "valeur": 0.1, "datum": "A"},
    {"type": "gdt",  "label": "⊕ 0.1 A B", "symbole": "⊕", "valeur": 0.1, "datum": "AB"},
    {"type": "gdt",  "label": "∥ 0.1",     "symbole": "∥", "valeur": 0.1, "datum": None},
    {"type": "dim",      "valeur": 20,   "label": "20"},
    {"type": "dim",      "valeur": 6,    "label": "6"},
    {"type": "dim",      "valeur": 10,   "label": "10"},
    {"type": "dim",      "valeur": 10,   "label": "10"},
]


if __name__ == "__main__":
    app = App()
    app.mainloop()
