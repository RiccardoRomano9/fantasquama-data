"""Applica le probabili formazioni a `serieA-base.json` e scrive `serieA.json`.

E' il pezzo che gira ogni due ore su GitHub Actions. Non tocca l'archivio dei
voti e non ne ha bisogno: la parte lenta -- le stime del modello -- e' gia'
dentro al file base, prodotto a mano quando arrivano voti nuovi. Qui cambia
solo cio' che cambia ogni settimana: chi gioca, chi e' in panchina, chi e'
fuori.

    python aggiorna.py                      # scarica e aggiorna in loco
    python aggiorna.py --probabili p.json   # usa un file gia' scaricato

**Non c'e' una seconda implementazione della formula.** Le costanti e le
funzioni sono quelle di `fantasquama.lineups`, copiate qui da `build.py`
insieme al resto: se cambiano di la', si rigenera. Riscriverle a mano sarebbe
il modo piu' sicuro di farle divergere proprio mentre nessuno guarda.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fantasquama import fixtures
from fantasquama import lineups
from fantasquama.scoring import EVENTS

ODDS_STALE_DISTANT_HOURS = 24
ODDS_STALE_NEAR_HOURS = 12
ODDS_STALE_MATCHDAY_HOURS = 3
ODDS_STALE_LIVE_HOURS = 2


def scarica(
    script: str, destinazione: Path, obbligatorio: bool = True, extra: list[str] | None = None
) -> Path | None:
    """Esegue uno degli `fetch_*` che stanno accanto a questo file."""
    percorso = Path(__file__).resolve().parent / script
    esito = subprocess.run([sys.executable, str(percorso), "-o", str(destinazione), *(extra or [])])
    if esito.returncode != 0:
        if obbligatorio:
            raise SystemExit(f"{script} non e' riuscito: nessun aggiornamento")
        print(f"  {script} non e' riuscito: si tiene quello che c'era")
        return None
    return destinazione


def aggiorna(
    base: dict, probabili: Path, quote: Path | None = None, now: datetime | None = None
) -> dict:
    """Il file base con la titolarita' rifatta secondo le probabili."""
    now = now or datetime.now(timezone.utc)
    giocatori = base["players"]
    rosa = pd.DataFrame({
        "listone_id": [p["id"] for p in giocatori],
        "player_name": [p["name"] for p in giocatori],
        "team": [p["team"] for p in giocatori],
        "role": [p["role"] for p in giocatori],
    })

    formazione, squadre = lineups.load_probabili(probabili)
    agganciata, persi = lineups.attach(formazione, rosa)
    if persi:
        print(f"  {len(persi)} nomi non sono in rosa, ignorati: {', '.join(persi)}")
    squadre = lineups.enrich_ballottaggi(squadre, rosa)

    per_id = {r.listone_id: r for r in agganciata.itertuples() if r.listone_id}
    slot = pd.Series([per_id[p["id"]].slot if p["id"] in per_id else None
                      for p in giocatori], dtype=object)
    pct = pd.Series([float(per_id[p["id"]].titolarita) if p["id"] in per_id else float("nan")
                     for p in giocatori], dtype=float)
    rank = pd.Series([p.get("penaltyRank") for p in giocatori], dtype=object)

    probabilita = pd.DataFrame({
        **{evento: [float(p["events"][evento]) for p in giocatori] for evento in EVENTS},
        "p_vote": [float(p["playProbability"]) for p in giocatori],
    })
    aggiornate = lineups.apply(
        probabilita,
        pd.Series([p["role"] for p in giocatori]),
        pd.Series([p["team"] for p in giocatori]),
        slot, pct, rank,
    )

    for i, giocatore in enumerate(giocatori):
        riga = per_id.get(giocatore["id"])
        giocatore["playProbability"] = round(float(aggiornate["p_vote"].iloc[i]), 3)
        for evento in ("rf", "rs"):
            giocatore["events"][evento] = round(float(aggiornate[evento].iloc[i]), 4)
        giocatore["lineupSlot"] = riga.slot if riga is not None else None
        giocatore["startingProbability"] = float(riga.titolarita) if riga is not None else None
        giocatore["status"] = (riga.stato or None) if riga is not None else None
        # La stima imparata non conosce le formazioni: se restasse quella di
        # prima, l'app ne farebbe la media con una che sa che il giocatore e'
        # infortunato, e il risultato non sarebbe ne' l'una ne' l'altra.
        giocatore["learnedPlayProbability"] = giocatore["playProbability"]

    aliases = lineups.roster.TEAM_ALIASES
    base["teams"] = [
        {
            "name": aliases.get(str(r.squadra), str(r.squadra)),
            "formation": str(r.modulo),
            "isOfficial": bool(r.ufficiale),
            "ballottaggi": r.ballottaggi,
        }
        for r in squadre.itertuples()
    ]
    quote_note = ""
    if quote is not None:
        aggiornate = aggiorna_quote(base, quote)
        if aggiornate:
            base["oddsUpdatedAt"] = now.isoformat(timespec="seconds")
            base["oddsSource"] = "the-odds-api"
            quote_note = f" Probabilita' partita aggiornate con le quote 1X2 ({aggiornate} partite)."
    base["generatedAt"] = now.isoformat(timespec="seconds")
    base["note"] = (
        base.get("baseNote", base.get("note", ""))
        + " Titolarita', panchina, ballottaggi e indisponibili aggiornati dalle probabili "
        + "formazioni del momento."
        + quote_note
    )
    return base


def riusa_quote(base: dict, precedente: dict | None) -> int:
    """Porta avanti le quote gia' scaricate per la stessa giornata."""
    if not precedente:
        return 0
    if base.get("season") != precedente.get("season") or base.get("gameweek") != precedente.get("gameweek"):
        return 0

    per_id = {
        player.get("id"): player
        for player in precedente.get("players", [])
        if player.get("id") is not None
    }
    riusate = 0
    for player in base["players"]:
        old = per_id.get(player.get("id"))
        if not old:
            continue
        stessa_partita = (
            player.get("team") == old.get("team")
            and player.get("opponent") == old.get("opponent")
            and player.get("home") == old.get("home")
        )
        if not stessa_partita or old.get("winProbability") is None:
            continue
        player["winProbability"] = old.get("winProbability")
        player["drawProbability"] = old.get("drawProbability")
        riusate += 1

    if riusate:
        base["oddsUpdatedAt"] = precedente.get("oddsUpdatedAt") or precedente.get("generatedAt")
        base["oddsSource"] = precedente.get("oddsSource", "the-odds-api")
    return riusate


def aggiorna_quote(base: dict, quote: Path) -> int:
    """Aggiorna le probabilita' partita dei giocatori dal CSV quote."""
    odds = fixtures._load_odds(quote)
    updated_matches: set[tuple[str, str]] = set()

    for player in base["players"]:
        team = fixtures._canonical(player.get("team", ""))
        opponent = fixtures._canonical(player.get("opponent", ""))
        if not opponent:
            continue
        if player.get("home") is True:
            key = (team, opponent)
            values = odds.get(key)
            win_index = 0
        else:
            key = (opponent, team)
            values = odds.get(key)
            win_index = 2
        if values is None:
            continue
        player["winProbability"] = round(float(values[win_index]), 3)
        player["drawProbability"] = round(float(values[1]), 3)
        updated_matches.add(key)
    return len(updated_matches)


def deve_scaricare_quote(base: dict, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    if not base.get("oddsUpdatedAt"):
        return True
    if not any(p.get("winProbability") is not None for p in base.get("players", [])):
        return True

    updated = _parse_time(base.get("oddsUpdatedAt"))
    if updated is None:
        return True
    hours = (now - updated).total_seconds() / 3600
    return hours >= _odds_stale_after_hours(base, now)


def _odds_stale_after_hours(base: dict, now: datetime) -> int:
    matches = [m for m in base.get("matches", []) if m.get("matchday") == base.get("gameweek")]
    if any(m.get("status") in {"IN_PLAY", "PAUSED", "LIVE"} for m in matches):
        return ODDS_STALE_LIVE_HOURS

    upcoming = [
        date for date in (_parse_time(m.get("date")) for m in matches)
        if date is not None and date >= now
    ]
    if not upcoming:
        return sys.maxsize

    next_match_hours = (min(upcoming) - now).total_seconds() / 3600
    if next_match_hours <= 12:
        return ODDS_STALE_MATCHDAY_HOURS
    if next_match_hours <= 72:
        return ODDS_STALE_NEAR_HOURS
    return ODDS_STALE_DISTANT_HOURS


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> None:
    qui = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=qui / "serieA-base.json")
    parser.add_argument("--probabili", type=Path, help="un file gia' scaricato")
    parser.add_argument("--odds", type=Path, help="un CSV quote gia' scaricato")
    parser.add_argument("--out", type=Path, default=qui / "serieA.json")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    probabili = args.probabili or scarica("fetch_lineups.py", qui / "probabili.json")
    # Le notizie non sono obbligatorie: un giornale che non risponde non deve
    # poter impedire di aggiornare chi gioca, che e' il motivo per cui l'app
    # esiste. Se saltano, restano quelle del giro precedente.
    notizie = scarica("fetch_news.py", qui / "news.json", obbligatorio=False)

    base = json.loads(args.base.read_text())
    base.setdefault("baseNote", base.get("note", ""))
    precedente = json.loads(args.out.read_text()) if args.out.exists() else None
    riusate = riusa_quote(base, precedente)
    if riusate:
        print(f"  quote precedenti riusate per {riusate} giocatori")
    quote = args.odds
    if quote is None:
        if deve_scaricare_quote(base, now):
            quote = scarica(
                "fetch_odds.py", qui / "odds-current.csv", obbligatorio=False,
                extra=["--base", str(args.base)],
            )
        else:
            print(f"  quote ancora fresche ({base.get('oddsUpdatedAt')}): niente chiamata a The Odds API")

    aggiornato = aggiorna(base, probabili, quote, now)
    if notizie is not None:
        aggiornato["news"] = json.loads(notizie.read_text())
    args.out.write_text(json.dumps(aggiornato, ensure_ascii=False, indent=1))
    print(f"{args.out}: {len(aggiornato['players'])} giocatori, "
          f"{len(aggiornato.get('news', []))} notizie, {args.out.stat().st_size:,} byte")


if __name__ == "__main__":
    main()
