#!/usr/bin/env python3
"""Associa URL dei ritratti Gazzetta usando nome e nascita da Wikidata.

Nessuna chiave, piano a pagamento o foto viene scaricata: Wikidata serve solo
per la data di nascita, Gazzetta resta la sorgente remota del ritratto.
"""
import argparse, json, re, time, unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen

from fantasquama.roster import name_tokens

WIKIDATA = "https://query.wikidata.org/sparql"
GAZZETTA = "https://images2.gazzettaobjects.it/assets-mc/calcio/giocatori"

def norm(value): return " ".join(name_tokens(value))

def slug(value):
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", plain).strip("_")

def query(names):
    values = " ".join(json.dumps(name) for name in names)
    sparql = f'''SELECT ?label ?birth WHERE {{
      VALUES ?wanted {{ {values} }}
      ?person rdfs:label ?label; wdt:P569 ?birth.
      FILTER(LANG(?label) IN ("it", "en"))
      FILTER(LCASE(STR(?label)) = LCASE(?wanted))
    }}'''
    url = f"{WIKIDATA}?{urlencode({'query': sparql, 'format': 'json'})}"
    request = Request(url, headers={"Accept": "application/sparql-results+json", "User-Agent": "FantaSquama/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=75) as response: return json.load(response)["results"]["bindings"]
        except (TimeoutError, URLError):
            if attempt == 2: raise
            time.sleep(3 * (attempt + 1))

def dates(players):
    names = sorted({p.get("fullName") or p.get("name", "") for p in players if p.get("fullName") or p.get("name")})
    out = {}
    # Query in gruppi piccoli: più gentile con il servizio pubblico e facilmente
    # diagnosticabile se un nome contiene caratteri insoliti.
    for start in range(0, len(names), 10):
        for row in query(names[start:start + 10]):
            key, date = norm(row["label"]["value"]), row["birth"]["value"][:10]
            out.setdefault(key, set()).add(date)
    return {key: next(iter(value)) for key, value in out.items() if len(value) == 1}

def gazzetta_url(name, date):
    year, month, day = date.split("-")
    return f"{GAZZETTA}/{slug(name.replace(' ', '_'))}_{day}{month}{year}.png"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args(); base = json.loads(args.base.read_text())
    known, changed, missing = dates(base.get("players", [])), 0, []
    for player in base.get("players", []):
        name = player.get("fullName") or player.get("name", "")
        date = known.get(norm(name))
        if not date: missing.append(player.get("name", "")); continue
        player["photoURL"], player["photoProviderID"] = gazzetta_url(name, date), f"gazzetta:{date}"
        changed += 1
    base["photosUpdatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.base.write_text(json.dumps(base, ensure_ascii=False, indent=1))
    print(f"{changed} URL Gazzetta associati; {len(missing)} fallback: {', '.join(missing[:12])}")

if __name__ == "__main__": main()
