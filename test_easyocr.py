#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test EasyOCR sur le plan bride (debug_ocr.png)"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import easyocr, os, sys

folder = os.path.dirname(os.path.abspath(__file__))
img = os.path.join(folder, "debug_ocr.png")

print("Chargement du modèle EasyOCR (fr+en)…")
print("(premier lancement = téléchargement ~100MB)")
reader = easyocr.Reader(["fr", "en"], gpu=False)

print(f"\nAnalyse de : {os.path.basename(img)}")
results = reader.readtext(img,
                           detail=1,
                           rotation_info=[90, 180, 270],
                           paragraph=False)

print(f"\n{'CONF':>5}  TEXTE")
print("-" * 40)
for (bbox, text, conf) in sorted(results, key=lambda x: -x[2]):
    print(f"{conf:5.2f}  {text}")

print(f"\nTotal : {len(results)} éléments détectés")
