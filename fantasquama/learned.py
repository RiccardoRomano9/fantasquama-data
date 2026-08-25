"""Stimatore addestrato: gli stessi output del modello a mano, imparati dai dati.

Il modello a mano moltiplica tre fattori -- giocatore, squadra, partita --
come se fossero indipendenti. Non lo sono: un attaccante forte, in una squadra
forte, contro una difesa debole, in casa, non e' il prodotto dei quattro
effetti. Un modello ad alberi quell'interazione la trova da solo.

Predice **eventi**, non fantapunti, esattamente come `estimate`: i punti
restano una funzione delle regole della lega, calcolata altrove. Predire i
punti direttamente cucirebbe dentro un regolamento solo e butterebbe via
l'unica funzione configurabile della V1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from fantasquama.features import HISTORY_COLUMNS
from fantasquama.scoring import EVENTS

# Tutto cio' che si sa prima del fischio d'inizio.
FIXTURE_FEATURES: tuple[str, ...] = ("home", "p_win", "p_draw", "p_lose", "advantage")
FEATURES: tuple[str, ...] = (*HISTORY_COLUMNS, *FIXTURE_FEATURES, "role_code")

TARGETS: tuple[str, ...] = ("voto", *EVENTS)


def build_features(history: pd.DataFrame, archive: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Matrice delle grandezze note prima della partita."""
    out = history[list(HISTORY_COLUMNS)].copy()
    out["home"] = pd.to_numeric(context["home"], errors="coerce").astype("float64")
    for name in ("p_win", "p_draw", "p_lose"):
        out[name] = pd.to_numeric(context[name], errors="coerce")
    out["advantage"] = out["p_win"] - out["p_lose"]
    roles = archive["role"].fillna("").astype(str).to_numpy()
    out["role_code"] = pd.Series(
        [{"P": 0, "D": 1, "C": 2, "A": 3}.get(r, -1) for r in roles], index=out.index, dtype="float64"
    )
    return out[list(FEATURES)]


@dataclass(frozen=True)
class LearnedEstimator:
    """Un classificatore per l'impiego, un regressore per ogni evento."""

    plays: HistGradientBoostingClassifier
    conditional: dict[str, HistGradientBoostingRegressor]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        matrix = features.to_numpy(np.float64)
        out = pd.DataFrame(index=features.index)
        out["p_vote"] = np.clip(self.plays.predict_proba(matrix)[:, 1], 0.0, 1.0)
        for name, model in self.conditional.items():
            out[name] = model.predict(matrix)
        # Le quantita' non possono essere negative; il voto sta nella sua scala.
        for name in EVENTS:
            out[name] = out[name].clip(lower=0.0)
        out["voto"] = out["voto"].clip(lower=1.0, upper=10.0)
        return out


def fit(features: pd.DataFrame, archive: pd.DataFrame, train_mask: np.ndarray, seed: int = 0) -> LearnedEstimator:
    """Addestra sulle sole righe di `train_mask`.

    Gli eventi si imparano dalle sole presenze: chiedere "quanti gol fa" a chi
    non e' sceso in campo insegnerebbe al modello che restare fuori significa
    non segnare, che e' vero e inutile -- lo dice gia' `p_vote`.
    """
    matrix = features.to_numpy(np.float64)
    played = archive["played"].to_numpy(bool)

    plays = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed
    )
    plays.fit(matrix[train_mask], played[train_mask])

    conditional: dict[str, HistGradientBoostingRegressor] = {}
    usable = train_mask & played
    for name in TARGETS:
        target = pd.to_numeric(archive[name], errors="coerce").to_numpy(np.float64)
        ok = usable & np.isfinite(target)
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed
        )
        model.fit(matrix[ok], target[ok])
        conditional[name] = model

    return LearnedEstimator(plays=plays, conditional=conditional)
