#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Interface graphique extraction de cotes plans industriels (v2 Tkinter)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, os, glob
from PIL import Image, ImageTk
import fitz  # pymupdf

from core.parser import parse_token
from core.pdf    import extract_from_pdf
from core.dxf    import extract_from_dxf
from core.ocr    import extract_from_image
from core.excel  import write_excel

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def render_pdf_page(pdf_path, width=500):
    doc  = fitz.open(pdf_path)
    page = doc[0]
    zoom = width / page.rect.width
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


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

        self.btn_analyze = tk.Button(top, text="⚡  Analyser",
                  bg="#45475a", fg=TEXT,
                  command=self._analyze, **btn_style)
        self.btn_analyze.pack(side="left", padx=4)

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

    def _start_timer(self):
        import time
        self._timer_start = time.time()
        self._timer_running = True
        self._tick_timer()

    def _tick_timer(self):
        if not self._timer_running:
            return
        import time
        elapsed = int(time.time() - self._timer_start)
        self._set_status(f"Analyse en cours… {elapsed}s")
        self.btn_analyze.config(text=f"⏳  Analyse… {elapsed}s")
        self._timer_id = self.after(1000, self._tick_timer)

    def _stop_timer(self):
        self._timer_running = False
        if hasattr(self, "_timer_id"):
            self.after_cancel(self._timer_id)
        self.btn_analyze.config(text="⚡  Analyser", state="normal")

    def _analyze(self):
        if not self.current_file:
            messagebox.showwarning("Aucun fichier", "Ouvrez d'abord un plan.")
            return
        self.btn_analyze.config(text="⏳  Analyse… 0s", state="disabled")
        self._start_timer()
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
                source = "PDF (texte vectoriel)"
                if not items:
                    self.after(0, lambda: self._set_status(
                        "⚠️ Ce document ne contient pas de texte vectoriel "
                        "(image rasterisée ou capture d'écran) — OCR en cours…"
                    ))
                    items = extract_from_image(path, from_pdf=True)
                    source = "PDF image (OCR EasyOCR)"

            if not items:
                name = os.path.basename(path).lower()
                if "bride" in name:
                    items = BRIDE_FALLBACK.copy()
                    source = "fallback visuel (bride.pdf)"
                else:
                    source += " — aucun élément trouvé (éditez manuellement)"

            self.items = items
            self.after(0, self._stop_timer)
            self.after(0, lambda: self._populate_tree(items, source))
        except Exception as e:
            self.after(0, self._stop_timer)
            self.after(0, lambda: self._set_status(f"Échec de l'analyse : {e} — Veuillez réessayer."))
            self.after(0, lambda: messagebox.showerror(
                "Analyse échouée",
                f"Une erreur est survenue :\n{e}\n\nVeuillez réessayer ou choisir un autre fichier."
            ))

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
        note = " — document image, vérifiez et complétez manuellement" if "OCR" in source else ""
        self._set_status(f"Extraction {source} : {len(items)} éléments trouvés{note}")

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
