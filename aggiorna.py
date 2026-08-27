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


def aggiorna(base: dict, probabili: Path, quote: Path | None = None) -> dict:
    """Il file base con la titolarita' rifatta secondo le probabili."""
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
            quote_note = f" Probabilita' partita aggiornate con le quote 1X2 ({aggiornate} partite)."
    base["generatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base["note"] = (
        base.get("baseNote", base.get("note", ""))
        + " Titolarita', panchina, ballottaggi e indisponibili aggiornati dalle probabili "
        + "formazioni del momento."
        + quote_note
    )
    return base


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


def main() -> None:
    qui = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=qui / "serieA-base.json")
    parser.add_argument("--probabili", type=Path, help="un file gia' scaricato")
    parser.add_argument("--odds", type=Path, help="un CSV quote gia' scaricato")
    parser.add_argument("--out", type=Path, default=qui / "serieA.json")
    args = parser.parse_args()

    probabili = args.probabili or scarica("fetch_lineups.py", qui / "probabili.json")
    quote = args.odds or scarica(
        "fetch_odds.py", qui / "odds-current.csv", obbligatorio=False,
        extra=["--base", str(args.base)],
    )
    # Le notizie non sono obbligatorie: un giornale che non risponde non deve
    # poter impedire di aggiornare chi gioca, che e' il motivo per cui l'app
    # esiste. Se saltano, restano quelle del giro precedente.
    notizie = scarica("fetch_news.py", qui / "news.json", obbligatorio=False)

    base = json.loads(args.base.read_text())
    base.setdefault("baseNote", base.get("note", ""))
    aggiornato = aggiorna(base, probabili, quote)
    if notizie is not None:
        aggiornato["news"] = json.loads(notizie.read_text())
    args.out.write_text(json.dumps(aggiornato, ensure_ascii=False, indent=1))
    print(f"{args.out}: {len(aggiornato['players'])} giocatori, "
          f"{len(aggiornato.get('news', []))} notizie, {args.out.stat().st_size:,} byte")


if __name__ == "__main__":
    main()
