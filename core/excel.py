import json, os
import xlrd, xlwt
from xlutils.copy import copy as xl_copy

def write_excel(items, xls_path):
    rb  = xlrd.open_workbook(xls_path, formatting_info=True)
    wb  = xl_copy(rb)
    ws  = wb.get_sheet(0)
    row = 14
    for item in items:
        ws.write(row, 1, item.get("label", ""))
        if item.get("type") == "gdt":
            obs = json.dumps(
                {"symbole": item.get("symbole", item.get("label", "")),
                 "valeur": item.get("valeur"), "cadre": True},
                ensure_ascii=False,
            )
            ws.write(row, 8, obs)
        row += 2
    out = xls_path.replace(".xls", "_rempli.xls")
    wb.save(out)
    return out
