"""
Step 2: Print the full column list per year and group columns into thematic domains.

Purpose: find which analytical topics are consistently measured 2012-2022,
since raw column NAMES are unstable but the underlying questionnaire topics are not.
"""

import json
import os
import re
from collections import defaultdict

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

with open(os.path.join(OUT_DIR, "schema.json"), encoding="utf-8") as fh:
    schema = json.load(fh)


def clean(c):
    return c.strip().strip('"').lower()


# Topic detection by name fragments. GHS renamed Q-codes (2012-18) to
# descriptive prefixes (2019-22), so match on both conventions.
TOPICS = {
    "identifiers/design": r"^(uqnr|personnr|psu|stratum|person_wgt|prov|metro|geotype|dc_|district)",
    "demographics": r"(gender|^sex$|^age|race|population|birth|language|marital|relsh|relationship|spouse|fath|moth|parent)",
    "education": r"(edu|hiedu|school|grade|q1(1[0-9]|[5-9])|attend|literac|wrtn|read|form|wrtl|^cal$|fet|learn|tuition|textbook|feeder)",
    "early childhood (ECD)": r"(ecd|chldatt|daycare|creche)",
    "health/healthcare": r"(health|hlth|illn|injur|medical|consult|hosp|clinic|chronic|hiv|tb_|smok|drink|alcohol|disab|see|hear|walk|remember|selfcare|communicat)",
    "disability/functioning": r"(disab|q1[45]_|difficult)",
    "labour market": r"(emp|work|job|unemp|occup|industr|labour|labor|lm_|wage|salar|earn|busine|hours|absen|seek)",
    "income/grants/welfare": r"(inc_|income|grant|pension|salar|remit|social|sassa|hunger|food|subsid)",
    "mortality/fertility": r"(death|died|mort|birth|fertil|preg)",
    "migration": r"(migr|moved|move|born_|prevprov|prev_)",
    "crime/safety": r"(crime|crim|safe|victim|assault|theft|murder|police)",
    "transport": r"(transport|travel|taxi|bus|train|commut|walk_time|distance)",
}


def topics_of(col):
    hits = [t for t, pat in TOPICS.items() if re.search(pat, col)]
    return hits or ["other/unclassified"]


# ---- per-year full column dump ----
for r in schema:
    cols = [clean(c) for c in r["columns"]]
    print(f"\n{'='*78}\n{r['year']}  ({r['n_cols']} cols, {r['n_rows']:,} rows, {r['encoding']})\n{'='*78}")
    print(", ".join(cols))

# ---- topic coverage matrix ----
years = [r["year"] for r in schema]
counts = defaultdict(dict)
for r in schema:
    for c in {clean(c) for c in r["columns"]}:
        for t in topics_of(c):
            counts[t][r["year"]] = counts[t].get(r["year"], 0) + 1

print(f"\n\n{'='*90}")
print("TOPIC COVERAGE: number of columns per topic per year")
print(f"{'='*90}")
hdr = f"{'topic':<26}" + "".join(f"{y:>6}" for y in years)
print(hdr)
print("-" * len(hdr))
order = list(TOPICS.keys()) + ["other/unclassified"]
for t in order:
    row = f"{t:<26}" + "".join(f"{counts[t].get(y, 0):>6}" for y in years)
    print(row)
