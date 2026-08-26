#!/usr/bin/env python3
"""Wikipedia/Wikidata in batch -> URL ritratti Gazzetta, con cache."""
import argparse, json, re, unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fantasquama.roster import name_tokens

GAZZETTA = "https://images2.gazzettaobjects.it/assets-mc/calcio/giocatori"

def api(url, **params):
    params.update(format="json", redirects=1)
    req = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "FantaSquama/1.0"})
    with urlopen(req, timeout=30) as response: return json.load(response)
def norm(v): return " ".join(name_tokens(v))
def slug(v):
    v = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", v).strip("_")
def photo_url(name, birth):
    y, m, d = birth.split("-")
    return f"{GAZZETTA}/{slug(name)}_{d}{m}{y}.png"

def births(names):
    out = {}
    for start in range(0, len(names), 20):
        chunk = names[start:start + 20]
        # Wikipedia risolve titolo->QID in una sola chiamata per venti nomi.
        pages = api("https://it.wikipedia.org/w/api.php", action="query", prop="pageprops", titles="|".join(chunk)).get("query", {}).get("pages", {})
        qids = {norm(page.get("title", "")): page.get("pageprops", {}).get("wikibase_item") for page in pages.values()}
        ids = [qid for qid in qids.values() if qid]
        if not ids: continue
        entities = api("https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(ids), props="claims").get("entities", {})
        by_qid = {}
        for qid in ids:
            claims = entities.get(qid, {}).get("claims", {}).get("P569", [])
            value = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "") if claims else ""
            if len(value) >= 11: by_qid[qid] = value[1:11]
        out.update({name: by_qid[qid] for name, qid in qids.items() if qid in by_qid})
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--base", type=Path, required=True); p.add_argument("--cache", type=Path, default=Path("player-birthdates.json")); a=p.parse_args()
    base=json.loads(a.base.read_text()); cache=json.loads(a.cache.read_text()) if a.cache.exists() else {}
    todo=[x.get("fullName") or x.get("name", "") for x in base["players"] if norm(x.get("fullName") or x.get("name", "")) not in cache]
    cache.update(births(todo))
    hit=0
    for player in base["players"]:
        name=player.get("fullName") or player.get("name", ""); birth=cache.get(norm(name))
        if birth: player["photoURL"], player["photoProviderID"] = photo_url(name,birth), f"gazzetta:{birth}"; hit+=1
    a.cache.write_text(json.dumps(cache,ensure_ascii=False,indent=1)); base["photosUpdatedAt"]=datetime.now(timezone.utc).isoformat(timespec="seconds"); a.base.write_text(json.dumps(base,ensure_ascii=False,indent=1)); print(f"{hit} URL Gazzetta associati")
if __name__ == "__main__": main()
