#links to ref.bib and replaces [@citation_example] with [1] etc

import re
import json
import sys
import os
import bibtexparser

#gets notebook from vscode
NB_PATH  = sys.argv[1] if len(sys.argv) > 1 else "referencestest.ipynb"
BIB_PATH = os.path.join(os.path.dirname(os.path.abspath(NB_PATH)), "ref.bib")

#gets bib file for refrences
if not os.path.exists(BIB_PATH):
    print(f" Could not find .bib file at: {BIB_PATH}")
    sys.exit(1)

with open(BIB_PATH, encoding="utf-8") as bibfile:
    bib_db = bibtexparser.load(bibfile)

bib_dict = {entry["ID"]: entry for entry in bib_db.entries}

# --- Load notebook ---
if not os.path.exists(NB_PATH):
    print(f" Could not find notebook at: {NB_PATH}")
    sys.exit(1)

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

citation_pattern = re.compile(r"\[@([^\]]+)\]")

# citation order for [1], [2], ... [n]
citation_order = []

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        for key in citation_pattern.findall(source):
            key = key.strip()
            if key not in citation_order:
                citation_order.append(key)

if not citation_order:
    print("No citations found — nothing to do.")
    sys.exit(0)

cite_map = {k: i + 1 for i, k in enumerate(citation_order)}

print(f"Found {len(citation_order)} unique citation(s):")
for k, n in cite_map.items():
    status = "✓" if k in bib_dict else "⚠  NOT IN BIB"
    print(f"  [{n}] {k}  {status}")

# replaces [@key] → [N] in markdown
def replace_citation(match):
    key = match.group(1).strip()
    return f"[{cite_map.get(key, '?')}]"

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"])
        new_text = citation_pattern.sub(replace_citation, text)
        cell["source"] = [new_text]

# numbered reference list 
ref_lines = ["## References\n\n"]

for key in citation_order:
    n = cite_map[key]
    entry = bib_dict.get(key)

    if entry is None:
        ref_lines.append(f"[{n}] **{key}** — *not found in ref.bib*\n\n")
        continue

    authors = entry.get("author", "Unknown authors").replace("\n", " ")
    title   = entry.get("title",  "Untitled").strip("{}")
    year    = entry.get("year",   "n.d.")
    journal = (
        entry.get("journal")
        or entry.get("booktitle")
        or entry.get("publisher")
        or ""
    ).strip("{}")

    line = f"[{n}] {authors} ({year}). *{title}*"
    if journal:
        line += f". {journal}."
    ref_lines.append(line + "\n\n")

# --- 4. Remove any existing References cell to avoid duplicates ---
nb["cells"] = [
    c for c in nb["cells"]
    if not (
        c["cell_type"] == "markdown"
        and "".join(c["source"]).startswith("## References")
    )
]

# --- 5. Append fresh References cell at the bottom ---
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ref_lines
})

# --- 6. Write updated notebook back to disk ---
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n Done — '{os.path.basename(NB_PATH)}' updated.")
print("   VS Code will prompt you to reload — click 'Revert File'.")
