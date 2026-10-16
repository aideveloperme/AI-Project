#!/usr/bin/env python3
"""Convert the Exeton product CSV into products.js for the app.

Usage:  python3 tools/csv_to_products.py data/exeton-products.csv > products.js
CSV columns: SL. No, URL, Descritpion (sic), Title. Optional: Price, Category.
"""
import csv, json, re, sys

CATEGORIES = [
    ("all", "All"),
    ("gpus", "GPUs"),
    ("ai-systems", "AI Systems"),
    ("servers", "Servers"),
    ("workstations", "Workstations & PCs"),
    ("storage", "Storage"),
    ("components", "Components"),
    ("networking", "Networking"),
    ("power-racks", "Power & Racks"),
    ("accessories", "Accessories"),
]

# Ordered rules: first match wins. Patterns are matched against the cleaned title (case-insensitive).
RULES = [
    ("networking", r"quantum-x|infiniband|connectx|switch"),
    ("power-racks", r"\bups\b|back-ups|\bpdu\b|rack ats|rectifier|battery|smartslot|rack cabinet|cabinet enclosure|inrow"),
    ("accessories", r"cable|carrying case|monitor|projector|anti-glare|adapter|bracket|termination tool|desk stand|front panel|cable manager"),
    ("gpus", r"tensor core gpu|graphics? card|^(nvidia|pny|gigabyte)\b(?!.*(server|system|spark|solution)).*\b(rtx|geforce|h[12]00 nvl)\b"),
    ("components", r"\bcpu\b(?! superchip)|server processor|^(?!.*(server|workstation)).*\bprocessor\b|xeon (gold |w9-)|\(\d+-core\)|motherboard|trx50|m13ses"),
    ("ai-systems", r"\bdgx\b(?!.*spark)|\bhgx\b|\bmgx\b|nvl72|nvl4|\bgb[23]00\b|\bb[23]00\b|gaudi|xe97|xe8712|sr680a|ai server|ai supercomputer|nvidia blackwell|g893|ai training"),
    ("servers", r"tower server|rack server|(rackmount|general purpose|compute|node) server|\d+u server(?! chassis)|server system|\(rtx pro"),
    ("workstations", r"workstation|tower|dgx spark|chromebox|chromebit|chromebook|thin client|cad "),
    ("storage", r"\bnas\b|jbod|storage|\bssd\b|\bhdd\b|hard drive|all-flash|tape|ai920|bd/dvd|nvme expansion|mobile rack"),
    ("components", r"\bddr[345]\b|lrdimm|rdimm|udimm|motherboard|server board|\bmicro atx\b|mini itx|lga 4677|trx50|m13ses|\bpsu\b|fan module|case fan|\bfans\b|af1[24]0|cold-plate|cooling|chassis|sound board|audio card|power supply|mini-converter rack"),
    ("gpus", r"geforce|\brtx\b|quadro|\bh[12]00\b|\bl40s?\b|\bl4\b|\ba(2|10|16|30|800)\b|\bt400\b|tensor core gpu|graphic"),
    ("servers", r"server|superserver|\bblade\b|proliant|thinksystem|poweredge|node|platform|cloud ?dc|hyper|flextwin|solution|x1[34]|esc\d"),
]

BRANDS = ["NVIDIA", "Supermicro", "GIGABYTE", "PNY", "ASUS", "HPE", "Lenovo", "Dell", "Intel", "AMD", "TensorEX",
          "Valence", "Crucial", "Kingston", "Micron", "ADATA", "Corsair", "APC", "Panduit", "AJA", "QNAP", "ASUSTOR",
          "3M", "3Dconnexion", "Belkin", "Belden", "Angelbird", "Acer", "Elo", "Pioneer", "Addonics", "AddOn", "3ware", "WD"]


def clean_title(t):
    t = re.sub(r"\s+", " ", t.replace("™", "").replace("®", "")).strip()
    t = re.sub(r"\s*\|\s*Buy .*$", "", t)              # "X | Buy X"
    t = re.sub(r"^Buy\s+", "", t)
    t = re.sub(r"\s+(Online|for Sale)\b.*$", "", t) if not t.endswith("Server") else re.sub(r"\s+Online$", "", t)
    t = re.sub(r"\s+for Sale\b", "", t)
    t = re.sub(r"\s+Online$", "", t)
    return t.strip(" -|")


def category(title, url):
    text = f"{title} {url.rsplit('/', 1)[-1].replace('-', ' ')}".lower()
    for cat, pat in RULES:
        if re.search(pat, title.lower()):
            return cat
    for cat, pat in RULES:
        if re.search(pat, text):
            return cat
    return "accessories"


def brand(title, url):
    low = f"{title} {url}".lower()
    if "poweredge" in low:
        return "Dell"
    if "proliant" in low:
        return "HPE"
    if "thinksystem" in low:
        return "Lenovo"
    for text in (title.lower(), low):  # earliest brand named in the title wins
        hits = [(m.start(), b) for b in BRANDS
                for m in [re.search(rf"(?<![a-z0-9]){re.escape(b.lower())}(?![a-z0-9])", text)] if m]
        if hits:
            return min(hits)[1]
    if re.search(r"\besc\d", low):
        return "ASUS"
    if re.search(r"\b(sys|ssg|as|asg|sbi|cse)-|superserver|a\+ server|x1[34]\b|hyper", low):
        return "Supermicro"
    if re.search(r"\b[bghx]\d{3}-|\b[gx]l?\w\d-", low):
        return "GIGABYTE"
    return None


def specs(title, desc, cat, b):
    s = {}
    if b:
        s["Brand"] = b
    mem = re.search(r"(\d{1,4})\s?(GB|TB)\b", title, re.I)
    if mem and cat in ("gpus", "components", "storage", "workstations", "ai-systems"):
        s["Memory" if cat in ("gpus", "ai-systems") or "DDR" in title.upper() else "Capacity"] = f"{mem.group(1)} {mem.group(2).upper()}"
    ddr = re.search(r"DDR\d(?:-\d{4})?", title, re.I)
    if ddr:
        s["Type"] = ddr.group(0).upper()
    gpus = re.search(r"(\d{1,2})x? ?(?:x )?GPU", title, re.I) or re.search(r"(\d{1,2})-GPU", title, re.I)
    if gpus and cat in ("servers", "ai-systems"):
        s["GPUs"] = gpus.group(1)
    if cat in ("gpus", "ai-systems", "workstations"):
        # Inferred from the model name only: some CSV descriptions name the wrong architecture.
        for arch, pat in [("Blackwell", r"\b(G?B[23]00|GB10|Blackwell|RTX (PRO|50\d0))\b"),
                          ("Hopper", r"\b(G?H[12]00)\b"),
                          ("Ada Lovelace", r"\b(Ada|L40S?|L4)\b"),
                          ("Ampere", r"\b(A(2|10|16|30|800)|RTX A\d{3,4})\b"),
                          ("Turing", r"\bT400\b")]:
            if re.search(pat, title, re.I if arch == "Blackwell" else 0):
                s["Architecture"] = arch
                break
    ru = re.search(r"\b(\d{1,2})U\b", title)
    if ru:
        s["Form factor"] = f"{ru.group(1)}U rackmount"
    elif re.search(r"tower", title, re.I):
        s["Form factor"] = "Tower"
    cpu = re.search(r"(Dual|Single|\dx)\s+(AMD EPYC|Intel Xeon)[\w /.-]*?(?= Series| processor|\s\d+U|\s-|$)", title, re.I)
    if cpu and cat in ("servers", "ai-systems", "storage"):
        s["Processors"] = cpu.group(0).strip()
    watts = re.search(r"(\d{2,4})\s?W\b", title)
    if watts:
        s["Power"] = f"{watts.group(1)} W"
    part = re.search(r"\(?\b((?:SYS|SSG|AS|ASG|SBI|CSE|TS\d|TTT|VW[SF])\s?-\s?[A-Z0-9-]+|9\d{2}-\d{5}-\d{4}-\d{3})\)?", title)
    if part:
        s["Part number"] = part.group(1).replace(" ", "")
    return s


def main(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    seen, products = set(), []
    for r in rows:
        url = (r.get("URL") or "").strip()
        raw = (r.get("Title") or "").strip()
        if not url or not raw:
            continue
        pid = url.rstrip("/").rsplit("/", 1)[-1]
        base, n = pid, 2
        while pid in seen:
            pid, n = f"{base}-{n}", n + 1
        seen.add(pid)
        title = clean_title(raw)
        desc = re.sub(r"\s+", " ", (r.get("Descritpion") or r.get("Description") or "")).strip()
        cat = (r.get("Category") or "").strip() or category(title, url)
        b = brand(title, url)
        price = (r.get("Price") or "").replace("$", "").replace(",", "").strip()
        products.append({
            "id": pid, "category": cat, "name": title, "brand": b,
            "price": float(price) if price else None,
            "summary": desc, "url": url, "specs": specs(title, desc, cat, b),
            "featured": int(r.get("SL. No") or 999) <= 16,
        })
    out = {"categories": [{"id": i, "name": n} for i, n in CATEGORIES], "products": products}
    sys.stdout.write("// Generated from data/exeton-products.csv by tools/csv_to_products.py. Do not edit by hand.\n")
    sys.stdout.write("window.CATALOG = " + json.dumps(out, indent=1, ensure_ascii=False) + ";\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/exeton-products.csv")
