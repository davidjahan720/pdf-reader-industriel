#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ezdxf, sys, os

dxf_path = "Bride montage general.dxf"
doc = ezdxf.readfile(dxf_path)

print("=== VERSION DXF ===", doc.dxfversion)
print("=== LAYOUTS ===")
for layout in doc.layouts:
    print(f"  '{layout.name}' — {len(list(layout))} entités")

print("\n=== TYPES D'ENTITÉS (modelspace) ===")
msp = doc.modelspace()
types = {}
for e in msp:
    t = e.dxftype()
    types[t] = types.get(t, 0) + 1
for t, n in sorted(types.items(), key=lambda x: -x[1]):
    print(f"  {t:20s} : {n}")

print("\n=== LAYERS ===")
for layer in doc.layers:
    print(f"  {layer.dxf.name}")

print("\n=== ECHANTILLON TEXT/MTEXT (20 premiers) ===")
count = 0
for e in msp:
    if e.dxftype() in ("TEXT", "MTEXT") and count < 20:
        try:
            txt = e.dxf.text if e.dxftype() == "TEXT" else e.text
            x = round(e.dxf.insert[0], 2) if hasattr(e.dxf, 'insert') else "?"
            y = round(e.dxf.insert[1], 2) if hasattr(e.dxf, 'insert') else "?"
            print(f"  [{e.dxftype()}] ({x},{y})  '{txt[:60]}'")
            count += 1
        except Exception as ex:
            pass

print("\n=== ECHANTILLON DIMENSION (20 premiers) ===")
count = 0
for e in msp:
    if e.dxftype() == "DIMENSION" and count < 20:
        try:
            val = e.dxf.get('actual_measurement', None)
            txt = e.dxf.get('text', '')
            dt  = e.dimtype & 0x0F
            print(f"  dimtype={dt}  val={val}  text='{txt}'")
            count += 1
        except Exception as ex:
            print(f"  ERREUR: {ex}")
