#!/usr/bin/env python3
"""Costruisce URL Gazzetta per i ritratti dei giocatori."""
import argparse, json, os, re, time, unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fantasquama.roster import name_tokens

API, SERIE_A = "https://v3.football.api-sports.io", 135
GAZZETTA = "https://images2.gazzettaobjects.it/assets-mc/calcio/giocatori"

def request(key, path, **params):
    url = f"{API}/{path}?{urlencode(params)}"
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"x-apisports-key": key, "User-Agent": "FantaSquama/1.0"}), timeout=30) as response:
                payload = json.load(response)
            if payload.get("errors"): raise SystemExit(f"API-Football: {payload['errors']}")
            return payload
        except HTTPError as error:
            if error.code != 429 or attempt == 3: raise
            time.sleep(12 * (attempt + 1))

def gazzetta_slug(name):
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", plain).strip("_")

def gazzetta_url(player):
    date = player.get("birth", {}).get("date")
    if not date: return None
    year, month, day = date.split("-")
    name = "_".join(filter(None, [player.get("firstname"), player.get("lastname")])) or player.get("name", "")
    return f"{GAZZETTA}/{gazzetta_slug(name)}_{day}{month}{year}.png"

def records(key, season):
    # Le prime tre pagine per rosa rispettano i limiti Free; ritentiamo i 429.
    teams = request(key, "teams", league=SERIE_A, season=season).get("response", [])
    out = []
    for entry in teams:
        team = entry.get("team", {}).get("id")
        if not team: continue
        for page in range(1, 4):
            payload = request(key, "players", team=team, season=season, page=page)
            out.extend(payload.get("response", []))
            if page >= int(payload.get("paging", {}).get("total", page)): break
            time.sleep(6.5)
        time.sleep(6.5)
    return out

def assign(base, source):
    by_name = {}
    for record in source:
        player = record.get("player", {})
        if gazzetta_url(player): by_name.setdefault(name_tokens(player.get("name", "")), []).append(player)
    changed, missing = 0, []
    for target in base.get("players", []):
        tokens = name_tokens(target.get("fullName") or target.get("name", ""))
        options = {str(p.get("id")): p for p in by_name.get(tokens, [])}
        if len(options) != 1: missing.append(str(target.get("name"))); continue
        player = next(iter(options.values()))
        target["photoURL"] = gazzetta_url(player)
        target["photoProviderID"] = f"gazzetta:{player['id']}"
        changed += 1
    return changed, missing

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True); parser.add_argument("--season", type=int, default=2024)
    args = parser.parse_args(); key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key: raise SystemExit("manca API_FOOTBALL_KEY: aggiungila come GitHub Secret, non nel repository")
    base = json.loads(args.base.read_text()); changed, missing = assign(base, records(key, args.season))
    base["photosUpdatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.base.write_text(json.dumps(base, ensure_ascii=False, indent=1))
    print(f"{changed} URL Gazzetta associati; {len(missing)} fallback: {', '.join(missing[:12])}")

if __name__ == "__main__": main()
