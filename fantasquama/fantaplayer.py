"""Storico stagionale di FantaPlayer come profilo lungo del giocatore.

I file in Downloads sono una riga per giocatore e stagione: non sostituiscono
l'archivio giornata per giornata (che e' piu' recente), ma danno al modello
un passato utile a inizio stagione e per chi rientra in Serie A. L'identita'
e' l'``Id`` del listone Fantacalcio.it, quindi il collegamento e' diretto e
non dipende da un confronto fragile fra nomi.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fantasquama.estimate import PREVIOUS_STATS

_FILE = re.compile(r"Stagione_(\d{4})_(\d{2})\.xlsx$", re.IGNORECASE)
_COLUMNS = {"Id", "R", "Pv", "Mv", "Gf", "Gs", "Rp", "R+", "R-", "Ass", "Amm", "Esp", "Au"}


def load(folder: Path) -> pd.DataFrame:
    """Legge i riepiloghi FantaPlayer e li porta a rate per presenza.

    ``Gf`` include il rigore segnato, come nell'archivio giornaliero; ``R+``
    viene quindi sottratto per non premiare una rete due volte.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(folder.glob("Statistiche_Fantacalcio_Stagione_*.xlsx")):
        found = _FILE.search(path.name)
        if not found:
            continue
        raw = pd.read_excel(path, sheet_name="Tutti", skiprows=1)
        missing = _COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f"{path.name}: colonne FantaPlayer mancanti {sorted(missing)}")
        raw = raw[raw["R"].astype("string").isin(["P", "D", "C", "A"])].copy()
        apps = pd.to_numeric(raw["Pv"], errors="coerce").fillna(0.0)
        raw = raw[apps > 0].copy()
        apps = apps[apps > 0]
        values = pd.DataFrame({
            "season": f"{found.group(1)}-{found.group(2)}",
            "listone_id": pd.to_numeric(raw["Id"], errors="raise").astype("int64").astype("string"),
            "role": raw["R"].astype("string"),
            "apps": apps.to_numpy(dtype=float),
            "vote_rate": np.minimum(1.0, apps.to_numpy(dtype=float) / 38.0),
            "voto": pd.to_numeric(raw["Mv"], errors="coerce").to_numpy(dtype=float),
        })
        for event, source in {
            "gf": "Gf", "gs": "Gs", "rp": "Rp", "rs": "R-", "rf": "R+",
            "ass": "Ass", "amm": "Amm", "esp": "Esp", "au": "Au",
        }.items():
            values[event] = pd.to_numeric(raw[source], errors="coerce").fillna(0.0).to_numpy(dtype=float) / values["apps"]
        values["gf"] -= values["rf"]
        # I riepiloghi non dichiarano quante porte inviolate ha fatto il
        # singolo portiere: inventarle da Gs stagionale sarebbe sbagliato.
        values["cs"] = np.nan
        frames.append(values[["season", "listone_id", "role", "apps", "vote_rate", *PREVIOUS_STATS]])
    if not frames:
        return pd.DataFrame(columns=["season", "listone_id", "role", "apps", "vote_rate", *PREVIOUS_STATS])
    return pd.concat(frames, ignore_index=True)


def enrich_previous(
    previous: pd.DataFrame, archive: pd.DataFrame, roster: pd.DataFrame, folder: Path
) -> pd.DataFrame:
    """Aggiunge al prior della stagione passata un profilo storico decadente.

    Le stagioni gia' nell'archivio giornaliero vengono ignorate: duplicarle
    raddoppierebbe la loro importanza. Ogni stagione meno recente pesa il 65%
    della successiva e l'intero storico contribuisce al massimo come sei
    presenze: e' un buon prior, non un modo per battere il presente.
    """
    if len(previous) != len(archive):
        raise ValueError("previous e archive devono avere la stessa lunghezza")
    history = load(folder)
    if history.empty:
        return previous

    known_seasons = set(archive["season"].astype(str))
    id_by_player = {
        str(row.player_id) if pd.notna(row.player_id) and str(row.player_id) else f"L{row.listone_id}": str(row.listone_id)
        for row in roster.itertuples()
    }
    out = previous.copy()
    for i, row in enumerate(archive.itertuples()):
        listone_id = id_by_player.get(str(row.player_id))
        if not listone_id:
            continue
        year = int(str(row.season)[:4])
        candidates = history[
            (history["listone_id"] == listone_id)
            & (~history["season"].isin(known_seasons))
            & (history["season"].str[:4].astype(int) < year)
        ].copy()
        if candidates.empty:
            continue
        age = year - candidates["season"].str[:4].astype(int)
        weights = candidates["apps"].to_numpy(float) * np.power(0.65, age.to_numpy(float) - 1)
        if weights.sum() <= 0:
            continue
        career = {name: float(np.average(candidates[name], weights=weights)) for name in (*PREVIOUS_STATS, "vote_rate")}
        support = min(6.0, max(1.0, float(np.sqrt(weights.sum()))))
        existing_raw = pd.to_numeric(out["apps_prev"].iloc[i], errors="coerce")
        existing = 0.0 if pd.isna(existing_raw) else float(existing_raw)
        for name in PREVIOUS_STATS:
            column = f"{name}_prev"
            current = pd.to_numeric(out[column].iloc[i], errors="coerce")
            if pd.isna(current) or existing <= 0:
                out.loc[out.index[i], column] = career[name]
            else:
                out.loc[out.index[i], column] = (float(current) * existing + career[name] * support) / (existing + support)
        current_rate = pd.to_numeric(out["vote_rate_prev"].iloc[i], errors="coerce")
        out.loc[out.index[i], "vote_rate_prev"] = (
            career["vote_rate"] if pd.isna(current_rate) or existing <= 0
            else (float(current_rate) * existing + career["vote_rate"] * support) / (existing + support)
        )
        out.loc[out.index[i], "apps_prev"] = existing + support
    return out
