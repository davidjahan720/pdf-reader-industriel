#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — API FastAPI pour extraction de cotes plans industriels
GPU activé si disponible (CUDA / NVIDIA A2)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import tempfile, os, shutil, json, re

from core.parser  import parse_token, iso_tolerance
from core.dxf     import extract_from_dxf
from core.pdf     import extract_from_pdf
from core.ocr     import extract_from_image
from core.excel   import write_excel

app = FastAPI(title="PDF Reader Industriel", version="2.0")

# ── Servir le frontend statique ──────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# ── Analyse d'un plan ────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    allowed = {".pdf", ".dxf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    if ext not in allowed:
        raise HTTPException(400, f"Format non supporté : {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if ext == ".dxf":
            items = extract_from_dxf(tmp_path)
            source = "DXF"
        elif ext == ".pdf":
            items = extract_from_pdf(tmp_path)
            source = "PDF (texte vectoriel)"
            if not items:
                items = extract_from_image(tmp_path, from_pdf=True)
                source = "PDF (OCR GPU)"
        else:
            items = extract_from_image(tmp_path)
            source = f"OCR image ({ext.lstrip('.')})"

        return {"source": source, "count": len(items), "items": items}
    finally:
        os.unlink(tmp_path)


# ── Export Excel ─────────────────────────────────────────────────────────────
@app.post("/export")
async def export(
    plan:  UploadFile = File(...),
    excel: UploadFile = File(...),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path  = os.path.join(tmpdir, plan.filename)
        excel_path = os.path.join(tmpdir, excel.filename)

        with open(plan_path,  "wb") as f: shutil.copyfileobj(plan.file,  f)
        with open(excel_path, "wb") as f: shutil.copyfileobj(excel.file, f)

        ext = os.path.splitext(plan.filename)[1].lower()
        if ext == ".dxf":
            items = extract_from_dxf(plan_path)
        elif ext == ".pdf":
            items = extract_from_pdf(plan_path) or extract_from_image(plan_path, from_pdf=True)
        else:
            items = extract_from_image(plan_path)

        out_path = write_excel(items, excel_path)
        return FileResponse(
            out_path,
            media_type="application/vnd.ms-excel",
            filename=os.path.basename(out_path),
        )


@app.get("/info")
async def info():
    import torch
    gpu = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if gpu else None
    return {"gpu": gpu, "gpu_name": name, "version": "2.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
