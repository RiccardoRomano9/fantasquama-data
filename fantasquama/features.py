"""Cio' che si sapeva di un giocatore PRIMA di ogni giornata.

Ogni colonna prodotta e' calcolata sulle sole giornate precedenti della stessa
stagione. E' la proprieta' che rende valido il backtest: se cade, il risultato
non significa niente. I test la verificano riga per riga.
"""

import numpy as np
import pandas as pd

from fantasquama.scoring import EVENTS

STAT_COLUMNS: tuple[str, ...] = ("voto", *EVENTS)

ROLLING_WINDOW: int = 5

HISTORY_COLUMNS: tuple[str, ...] = (
    "gw_elapsed",
    "apps_before",
    "apps_last5",
    "vote_rate",
    "team_goals_rate",
    *(f"{name}_mean" for name in STAT_COLUMNS),
)


def _team_goals_rate(ordered: pd.DataFrame) -> pd.Series:
    """Reti per giornata della squadra, sulle sole giornate precedenti.

    Un centrocampista dell'Inter e uno del Lecce possono avere la stessa
    storia personale e non la stessa probabilita' di bonus: senza questo il
    modello li tratta identici. Rigori inclusi -- sono reti della squadra a
    tutti gli effetti.

    Si passa per l'aggregato (stagione, squadra, giornata) e si torna indietro
    con un merge, non con un groupby sulle righe giocatore: sommare per
    giornata prima di mediare e' l'unico modo di avere reti *per giornata*
    invece che per giocatore-giornata.
    """
    reti = pd.to_numeric(ordered["gf"], errors="coerce").fillna(0.0) + pd.to_numeric(
        ordered["rf"], errors="coerce"
    ).fillna(0.0)
    per_gw = (
        ordered.assign(_reti=reti)
        .groupby(["season", "team", "gameweek"], as_index=False, observed=True)["_reti"]
        .sum()
        .sort_values(["season", "team", "gameweek"])
    )
    key = [per_gw["season"], per_gw["team"]]
    precedenti = per_gw["_reti"].groupby(key).shift(1).fillna(0.0).groupby(key).cumsum()
    giornate = per_gw.groupby(key).cumcount()
    per_gw["team_goals_rate"] = precedenti / giornate

    merged = ordered[["season", "team", "gameweek"]].merge(
        per_gw[["season", "team", "gameweek", "team_goals_rate"]],
        on=["season", "team", "gameweek"],
        how="left",
    )
    # il merge azzera l'indice: lo rimettiamo posizionalmente, come ovunque
    return pd.Series(merged["team_goals_rate"].to_numpy(), index=ordered.index)


def rolling_history(df: pd.DataFrame) -> pd.DataFrame:
    """Storia cumulata per giocatore, sfasata di una giornata.

    Ritorna un frame con lo stesso ordine di righe e lo stesso indice di `df`,
    anche quando l'indice di `df` contiene etichette ripetute: l'ordinamento
    e il ripristino sono posizionali (`iloc`), non basati sull'etichetta
    (`loc`), che con etichette duplicate espanderebbe le righe.
    """
    positions = np.lexsort((
        df["gameweek"].to_numpy(),
        df["player_id"].to_numpy(),
        df["season"].to_numpy(),
    ))
    ordered = df.iloc[positions]
    key = [ordered["season"], ordered["player_id"]]

    out = pd.DataFrame(index=ordered.index)
    out["gw_elapsed"] = ordered.groupby(key).cumcount().astype("float64")

    played = ordered["played"].astype("float64")
    out["apps_before"] = played.groupby(key).shift(1).fillna(0.0).groupby(key).cumsum()

    # Presenze nelle ultime ROLLING_WINDOW giornate, sempre escludendo quella
    # corrente: e' la differenza fra il cumulato di adesso e quello di cinque
    # giornate fa. E' una misura di recenza, mentre `apps_before` satura: chi
    # ha giocato le prime tre giornate e poi e' sparito supera ancora, sul
    # cumulato, chi ha giocato le ultime cinque.
    out["apps_last5"] = out["apps_before"] - (
        out["apps_before"].groupby(key).shift(ROLLING_WINDOW).fillna(0.0)
    )

    for name in STAT_COLUMNS:
        # gli eventi contano solo nelle giornate in cui il giocatore ha preso voto
        values = (pd.to_numeric(ordered[name], errors="coerce") * played).fillna(0.0)
        total = values.groupby(key).shift(1).fillna(0.0).groupby(key).cumsum()
        out[f"{name}_mean"] = total / out["apps_before"]

    out["vote_rate"] = out["apps_before"] / out["gw_elapsed"]
    out["team_goals_rate"] = _team_goals_rate(ordered)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.iloc[np.argsort(positions)][list(HISTORY_COLUMNS)]
