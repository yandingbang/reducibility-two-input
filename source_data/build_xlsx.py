# -*- coding: utf-8 -*-
"""Assemble the Source Data CSVs of export_source_data.py into one workbook per figure.

csv/<figure>/<panel>.csv  ->  Source_Data_<figure>.xlsx, one sheet per panel (sheet name = panel).
Note lines ('# ...') become the first rows of the sheet.  Needs only openpyxl (no numpy),
so it can run in any interpreter that has it: python build_xlsx.py
"""
import csv
import glob
import os

try:
    from openpyxl import Workbook
except ImportError:
    raise SystemExit("openpyxl is needed: pip install openpyxl")

HERE = os.path.dirname(os.path.abspath(__file__))


def cell(s):
    try:
        v = float(s)
        return int(v) if v.is_integer() and "." not in s and "e" not in s.lower() else v
    except ValueError:
        return s


for fig_dir in sorted(glob.glob(os.path.join(HERE, "csv", "*"))):
    fig = os.path.basename(fig_dir)
    wb = Workbook()
    wb.remove(wb.active)
    for path in sorted(glob.glob(os.path.join(fig_dir, "*.csv"))):
        ws = wb.create_sheet(os.path.basename(path)[:-4][:31])
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if row and row[0].startswith("# "):
                    ws.append([row[0][2:]])
                else:
                    ws.append([cell(v) for v in row])
    out = os.path.join(HERE, "Source_Data_%s.xlsx" % fig)
    wb.save(out)
    print("saved", out, "sheets:", wb.sheetnames)
