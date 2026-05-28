from .parser import parse_token

def extract_from_dxf(dxf_path):
    import ezdxf
    items = []
    doc = ezdxf.readfile(dxf_path)
    for e in doc.modelspace():
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
