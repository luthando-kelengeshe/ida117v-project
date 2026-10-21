"""
Step 1: Schema + volume scan across all GHS person-level files (2012-2022).

Reads only headers and streams rows to count records, so memory stays low.
Writes a machine-readable schema map to analysis/out/schema.json.
"""

import csv
import glob
import json
import os
import re
from collections import Counter

DATA_DIR = r"c:\Users\USER-PC\TUT\IDA\Project\ida117v-project\dataset"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)


def detect_encoding(path):
    """GHS files from Stats SA are usually cp1252; verify by trying strict decodes."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                for _ in range(5000):
                    if not fh.readline():
                        break
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def year_of(path):
    m = re.search(r"ghs-(\d{4})-", os.path.basename(path))
    return int(m.group(1))


def scan(path):
    enc = detect_encoding(path)
    with open(path, "r", encoding=enc, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        nrows = 0
        # sample values for a few key columns to judge label vs numeric coding
        for _ in reader:
            nrows += 1
    return {
        "file": os.path.basename(path),
        "year": year_of(path),
        "encoding": enc,
        "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
        "n_cols": len(header),
        "n_rows": nrows,
        "columns": header,
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")), key=year_of)
    results = [scan(f) for f in files]

    print(f"{'year':<6}{'rows':>10}{'cols':>7}{'MB':>9}  encoding")
    print("-" * 50)
    for r in results:
        print(
            f"{r['year']:<6}{r['n_rows']:>10,}{r['n_cols']:>7}"
            f"{r['size_mb']:>9.1f}  {r['encoding']}"
        )
    print("-" * 50)
    print(f"{'TOTAL':<6}{sum(r['n_rows'] for r in results):>10,}")

    # How stable are column names across years (case-insensitive)?
    freq = Counter()
    for r in results:
        for c in {c.strip().strip('"').lower() for c in r["columns"]}:
            freq[c] += 1
    n_years = len(results)
    print(f"\nColumn-name stability (case-insensitive) across {n_years} years:")
    for k in range(n_years, 0, -1):
        cols = sorted(c for c, v in freq.items() if v == k)
        if cols:
            print(f"  present in {k:>2} year(s): {len(cols):>4} columns")

    universal = sorted(c for c, v in freq.items() if v == n_years)
    print(f"\nColumns present in ALL {n_years} years ({len(universal)}):")
    print("  " + ", ".join(universal))

    with open(os.path.join(OUT_DIR, "schema.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {os.path.join(OUT_DIR, 'schema.json')}")


if __name__ == "__main__":
    main()
