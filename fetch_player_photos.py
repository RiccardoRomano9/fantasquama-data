#!/usr/bin/env python3
"""Associa i ritratti API-Football ai giocatori di un bundle pubblico."""
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fantasquama.roster import name_tokens

ENDPOINT, SERIE_A = "https://v3.football.api-sports.io", 135
TEAM_WORDS = {"ac", "as", "fc", "ssc", "us", "calcio", "football", "club", "1919"}
TEAM_ALIASES = {"hellas verona": "verona", "inter milan": "inter", "ac milan": "milan"}

def team_key(name):
    key = " ".join(w for w in name_tokens(name) if w not in TEAM_WORDS)
    return TEAM_ALIASES.get(key, key)

def request(key, path, **params):
    url = f"{ENDPOINT}/{path}?{urlencode(params)}"
    with urlopen(Request(url, headers={"x-apisports-key": key, "User-Agent": "FantaSquama/1.0"}), timeout=30) as response:
        payload = json.load(response)
    if payload.get("errors"): raise SystemExit(f"API-Football: {payload['errors']}")
    return payload

def load_all(key, season):
    # L'account Free blocca la quarta pagina della lega. La rosa di una
    # singola squadra sta invece nelle prime pagine: 20 piccole richieste
    # sono più affidabili e restano sotto le 100 richieste/giorno gratuite.
    teams = request(key, "teams", league=SERIE_A, season=season).get("response", [])
    out = []
    for entry in teams:
        team_id = entry.get("team", {}).get("id")
        if not team_id: continue
        page = 1
        while True:
            payload = request(key, "players", team=team_id, season=season, page=page)
            out.extend(payload.get("response", []))
            # API-Football Free non autorizza mai `page=4`, neppure per le
            # rose più grandi. Le prime tre contengono comunque la rosa
            # principale; gli eventuali assenti useranno il fallback.
            if page >= min(3, int(payload.get("paging", {}).get("total", page))): break
            page += 1
    return out
"""  # codice storico mantenuto fuori dal modulo per documentare il limite Free
    out, page = [], 1
    while True:
        url = f"{ENDPOINT}/players?{urlencode({'league': SERIE_A, 'season': season, 'page': page})}"
        with urlopen(Request(url, headers={"x-apisports-key": key, "User-Agent": "FantaSquama/1.0"}), timeout=30) as response:
            payload = json.load(response)
        if payload.get("errors"): raise SystemExit(f"API-Football: {payload['errors']}")
        out.extend(payload.get("response", []))
        if page >= int(payload.get("paging", {}).get("total", page)): return out
        page += 1
"""

def assign(base, records):
    candidates, by_name = {}, {}
    for record in records:
        player = record.get("player", {})
        for stats in record.get("statistics", []):
            key = (team_key(stats.get("team", {}).get("name", "")), name_tokens(player.get("name", "")))
            if all(key) and player.get("photo"):
                candidates.setdefault(key, []).append(player)
                by_name.setdefault(key[1], []).append(player)
    changed, missing = 0, []
    for target in base.get("players", []):
        team, tokens = team_key(target.get("team", "")), name_tokens(target.get("fullName") or target.get("name", ""))
        options = candidates.get((team, tokens), [])
        # Con il piano gratuito la stagione più recente disponibile è la
        # 2024: chi nel frattempo ha cambiato squadra non va perso se il suo
        # nome nell'archivio è univoco.
        if not options:
            options = by_name.get(tokens, [])
        if not options:
            options = [p for (candidate_team, candidate_tokens), values in candidates.items()
                       if candidate_team == team and (set(tokens) <= set(candidate_tokens) or set(candidate_tokens) <= set(tokens)) for p in values]
        unique = {str(p.get("id")): p for p in options}
        if len(unique) != 1: missing.append(str(target.get("name"))); continue
        player = next(iter(unique.values()))
        target["photoURL"], target["photoProviderID"] = player["photo"], player["id"]
        changed += 1
    return changed, missing

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True); parser.add_argument("--season", type=int, default=2024)
    args = parser.parse_args(); key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key: raise SystemExit("manca API_FOOTBALL_KEY: aggiungila come GitHub Secret, non nel repository")
    base = json.loads(args.base.read_text()); changed, missing = assign(base, load_all(key, args.season))
    base["photosUpdatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.base.write_text(json.dumps(base, ensure_ascii=False, indent=1))
    print(f"{changed} foto aggiornate; {len(missing)} senza aggancio: {', '.join(missing[:12])}")

if __name__ == "__main__": main()
