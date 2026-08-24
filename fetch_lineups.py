#!/usr/bin/env python3
"""Probabili formazioni da sosfanta.com: titolari, panchina, indisponibili.

    python fetch_lineups.py -o data/probabili.json
    python fetch_lineups.py --table

Sta fuori dal pacchetto come fetch_fixtures.py: `fantasquama/` non fa
chiamate di rete, cosi' il backtest e' riproducibile e non dipende da un
sito raggiungibile. Solo stdlib, nessuna dipendenza.

Le percentuali sono quelle della fonte e non sono probabilita' di prendere
voto: dicono quanto e' probabile che scenda in campo dal primo minuto. La
conversione la fa `fantasquama.lineups`, con una costante misurata.

**Il parsing dipende dall'HTML di un sito che non controlliamo.** Quando
cambia, `parse` torna zero partite e lo dice: e' l'unico modo di accorgersene,
perche' un file vecchio non ha nessun sintomo.
"""
import argparse
import html as htmllib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone

URL = "https://www.sosfanta.com/lista-formazioni/probabili-formazioni-serie-a/"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36")


def fetch(url: str = URL) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


# <span class="...bg-green-600">100%</span><span ...>Martinez Jo.</span>
ROW = re.compile(
    r'<span[^>]*bg-(?:green|amber|red)-\d+[^>]*>\s*(?P<pct>\d+)\s*%.*?'
    r'truncate">\s*(?P<name>[^<]+?)\s*</span>',
    re.S,
)


def parse(page: str) -> dict:
    out = {
        "fonte": "sosfanta.com — probabili formazioni serie a",
        "url": URL,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "giornata": None,
        "partite": [],
    }
    m = re.search(r"<time\s+datetime=\"([^\"]+)\"", page)
    if m:
        out["giornata"] = htmllib.unescape(m.group(1))

    cards = re.split(r'(?=<article\s+id="match-)', page)
    for card in cards[1:]:
        mid = re.search(r'id="match-([a-z-]+)"', card).group(1)
        date_m = re.search(r"<time\s+datetime=\"([^\"]+)\"", card)
        team1 = re.search(r'<h2[^>]*>([^<]+)</h2>', card)
        # secondo h2 = squadra ospite
        h2s = re.findall(r'<h2[^>]*>([^<]+)</h2>', card)
        forms = re.findall(r'text-primary">\s*([\d-]+)\s*<', card)
        partita = {
            "id": mid,
            "data": date_m.group(1) if date_m else None,
            "casa": {"squadra": htmllib.unescape(h2s[0]) if h2s else None,
                     "formazione": forms[0] if len(forms) > 0 else None,
                     "titolari": [], "panchina": [], "indisponibili": []},
            "ospite": {"squadra": htmllib.unescape(h2s[1]) if len(h2s) > 1 else None,
                       "formazione": forms[1] if len(forms) > 1 else None,
                       "titolari": [], "panchina": [], "indisponibili": []},
        }

        for h3 in re.finditer(
            r'<h3[^>]*>\s*(Titolari|Panchina|Indisponibili)\s*</h3>'
            r'(?P<body>.*?)(?=<h3|</article)',
            card,
            re.S,
        ):
            section = h3.group(1).lower()
            body = h3.group("body")
            if section == "indisponibili":
                # Come le altre sezioni: il sito mette casa e ospite in due
                # <ul>. Leggere l'intera sezione in blocco attribuiva alla
                # squadra di casa anche gli infortunati dell'altra -- e non si
                # notava, perche' i nomi sono comunque nomi.
                for i, col in enumerate(re.findall(r'<ul.*?</ul>', body, re.S)):
                    side = partita["casa"] if i == 0 else partita["ospite"]
                    items = re.findall(
                        r'truncate">\s*([^<]+?)\s*</span>'
                        r'(.*?)(?=<li|</ul>)',
                        col,
                        re.S,
                    )
                    for nome, resto in items:
                        stato = "infortunato" if "Infortunato" in resto else (
                            "squalificato" if "Squalificato" in resto else "indisponibile")
                        out_row = {"giocatore": htmllib.unescape(nome), "stato": stato}
                        note_m = re.search(r'leading-5[^>]*>\s*([^<]+?)\s*<', resto)
                        if note_m:
                            out_row["nota"] = htmllib.unescape(note_m.group(1))
                        side["indisponibili"].append(out_row)
            else:
                # il sito divide casa/ospite in due <ul>: prima colonna = casa
                cols = re.findall(r'<ul.*?</ul>', body, re.S)
                for i, col in enumerate(cols):
                    side = partita["casa"] if i == 0 else partita["ospite"]
                    for rm in ROW.finditer(col):
                        side[section].append({
                            "giocatore": htmllib.unescape(rm.group("name")).strip(),
                            "titolarita_pct": int(rm.group("pct")),
                        })
        out["partite"].append(partita)
    return out


def to_table(data: dict) -> str:
    lines = [f"Giornata: {data['giornata']}  (scraped {data['scraped_at_utc']})", ""]
    for p in data["partite"]:
        lines.append(
            f"== {p['casa']['squadra']}  vs  {p['ospite']['squadra']}"
            f"  |  {p['data'] or ''}"
        )
        for chi in ("casa", "ospite"):
            s = p[chi]
            lines.append(f"  {s['squadra']} ({chi.upper()}) — modulo {s['formazione'] or 'n/d'}:")
            lines.append(f"    Titolari ({len(s['titolari'])}):")
            for r in s["titolari"]:
                lines.append(f"      {r['titolarita_pct']:>3}%  {r['giocatore']}")
            lines.append(f"    Panchina ({len(s['panchina'])}):")
            for r in s["panchina"]:
                lines.append(f"      {r['titolarita_pct']:>3}%  {r['giocatore']}")
            if s["indisponibili"]:
                lines.append(f"    Indisponibili ({len(s['indisponibili'])}):")
                for r in s["indisponibili"]:
                    nota = f"  — {r['nota']}" if r.get("nota") else ""
                    lines.append(f"      [{r['stato']}] {r['giocatore']}{nota}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", help="salva JSON in questo file")
    ap.add_argument("--table", action="store_true", help="stampa tabella invece di JSON")
    args = ap.parse_args()

    data = parse(fetch())
    if not data["partite"]:
        print("WARNING: nessuna partita trovata (layout cambiato? fuori stagione?)",
              file=sys.stderr)

    payload = json.dumps(data, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"OK: {len(data['partite'])} partite -> {args.out}", file=sys.stderr)
    elif args.table:
        print(to_table(data))
    else:
        print(payload)


if __name__ == "__main__":
    main()
