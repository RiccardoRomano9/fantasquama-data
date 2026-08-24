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

from fantasquama import lineups
from fantasquama.scoring import EVENTS


def scarica(destinazione: Path) -> Path:
    """Le probabili del momento, con lo scraper che sta accanto a questo file."""
    script = Path(__file__).resolve().parent / "fetch_lineups.py"
    subprocess.run([sys.executable, str(script), "-o", str(destinazione)], check=True)
    return destinazione


def aggiorna(base: dict, probabili: Path) -> dict:
    """Il file base con la titolarita' rifatta secondo le probabili."""
    giocatori = base["players"]
    rosa = pd.DataFrame({
        "listone_id": [p["id"] for p in giocatori],
        "player_name": [p["name"] for p in giocatori],
        "team": [p["team"] for p in giocatori],
    })

    formazione, squadre = lineups.load_probabili(probabili)
    agganciata, persi = lineups.attach(formazione, rosa)
    if persi:
        print(f"  {len(persi)} nomi non sono in rosa, ignorati: {', '.join(persi)}")

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
        {"name": aliases.get(str(r.squadra), str(r.squadra)), "formation": str(r.modulo)}
        for r in squadre.itertuples()
    ]
    base["generatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base["note"] = (
        base.get("baseNote", base.get("note", ""))
        + " Titolarita', panchina e indisponibili aggiornati dalle probabili "
        + "formazioni del momento."
    )
    return base


def main() -> None:
    qui = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=qui / "serieA-base.json")
    parser.add_argument("--probabili", type=Path, help="un file gia' scaricato")
    parser.add_argument("--out", type=Path, default=qui / "serieA.json")
    args = parser.parse_args()

    probabili = args.probabili or scarica(qui / "probabili.json")
    base = json.loads(args.base.read_text())
    base.setdefault("baseNote", base.get("note", ""))
    aggiornato = aggiorna(base, probabili)
    args.out.write_text(json.dumps(aggiornato, ensure_ascii=False, indent=1))
    print(f"{args.out}: {len(aggiornato['players'])} giocatori, "
          f"{args.out.stat().st_size:,} byte")


if __name__ == "__main__":
    main()
