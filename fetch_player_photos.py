#!/usr/bin/env python3
"""Trova la nascita in Wikidata, la cachea, e costruisce gli URL Gazzetta."""
import argparse, json, re, time, unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fantasquama.roster import name_tokens

WIKIDATA = "https://www.wikidata.org/w/api.php"
GAZZETTA = "https://images2.gazzettaobjects.it/assets-mc/calcio/giocatori"

def norm(value): return " ".join(name_tokens(value))
def slug(value):
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")

def api(**params):
    params.update(format="json", origin="*")
    request = Request(f"{WIKIDATA}?{urlencode(params)}", headers={"User-Agent": "FantaSquama/1.0"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=25) as response: return json.load(response)
        except Exception:
            if attempt == 3: return {}
            time.sleep(1.5 * (attempt + 1))

def birthdate(name):
    # Una ricerca piccola per nome evita timeout SPARQL. Conserviamo il QID e
    # la data nel file cache, quindi questo costo esiste solo al primo giro.
    found = api(action="wbsearchentities", search=name, language="it", uselang="it", type="item", limit=5)
    ids = [row["id"] for row in found.get("search", []) if "football" in row.get("description", "").lower() or "calciatore" in row.get("description", "").lower()]
    if not ids: ids = [row["id"] for row in found.get("search", [])[:1]]
    if not ids: return None
    entities = api(action="wbgetentities", ids="|".join(ids), props="claims").get("entities", {})
    dates = []
    for qid in ids:
        claims = entities.get(qid, {}).get("claims", {}).get("P569", [])
        if claims:
            value = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "")
            if len(value) >= 11: dates.append((qid, value[1:11]))
    return dates[0] if len(dates) == 1 else None

def gazzetta_url(name, date):
    year, month, day = date.split("-")
    return f"{GAZZETTA}/{slug(name)}_{day}{month}{year}.png"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("player-birthdates.json"))
    args = parser.parse_args(); base = json.loads(args.base.read_text())
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    changed, missing = 0, []
    for player in base.get("players", []):
        name, key = player.get("fullName") or player.get("name", ""), norm(player.get("fullName") or player.get("name", ""))
        entry = cache.get(key)
        if entry is None:
            result = birthdate(name)
            entry = {"qid": result[0], "birthDate": result[1]} if result else False
            cache[key] = entry
            time.sleep(0.18)
        if not entry: missing.append(player.get("name", "")); continue
        player["photoURL"] = gazzetta_url(name, entry["birthDate"])
        player["photoProviderID"] = f"gazzetta:{entry['qid']}"
        changed += 1
    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    base["photosUpdatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.base.write_text(json.dumps(base, ensure_ascii=False, indent=1))
    print(f"{changed} URL Gazzetta associati; {len(missing)} fallback")

if __name__ == "__main__": main()
