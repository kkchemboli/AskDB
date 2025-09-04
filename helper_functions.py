import re, tempfile,os
def save_if_html(text: str, query: str):
    """Check if text is HTML and save to a temp file, else print."""
    if text.startswith("<!DOCTYPE html>") or text.startswith("<html"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
            tmp.write(text)
            print(f"✅ Plotly chart saved to temporary file: {tmp.name}")
        return tmp.name
    else:
        print(text)
        return None
        

def get_text_columns(db, table):
    """Return all text-like columns in a table."""
    rows = db._execute(f"PRAGMA table_info({table});")
    return [
        r["name"] for r in rows
        if "CHAR" in r["type"].upper() or "TEXT" in r["type"].upper()
    ]

def collect_unique_values(db, strip_numbers=False):
    """
    Collect all unique proper nouns from all text columns in the database.
    Returns them as lowercase strings for case-insensitive matching.
    """
    all_values = set()
    tables = db.get_usable_table_names()

    for table in tables:
        text_cols = get_text_columns(db, table)
        for col in text_cols:
            rows = db._execute(f'SELECT DISTINCT "{col}" FROM "{table}"')
            for r in rows:
                v = r[col]
                if isinstance(v, str):
                    v = v.strip()
                    if strip_numbers:
                        v = re.sub(r"\b\d+\b", "", v).strip()
                    # ignore pure numbers
                    if not re.fullmatch(r"\d+(\.\d+)?", v):
                        # accept strings starting with any letter
                        if re.match(r"^[A-Za-z][\w\s&'-]+$", v):
                            all_values.add(v)

    return list(all_values)
