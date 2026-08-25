"""Stima del voto base a partire da statistiche oggettive.

Il voto e' un giudizio di un giornalista: nessuna API lo fornisce, e l'app non
lo avra' mai. Qui lo approssimiamo con una retta per ruolo, tarata sui voti
storici. I voti servono solo a ricavare i coefficienti: dopo la taratura
l'archivio non e' piu' necessario, e nessun voto storico finisce nel prodotto.

Le grandezze usate sono solo quelle che l'app avra' davvero a runtime.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

CALIBRATION_FEATURES: tuple[str, ...] = (
    "gf_mean",
    "rf_mean",
    "ass_mean",
    "amm_mean",
    "esp_mean",
)

MIN_SAMPLES = 30
DEFAULT_VOTE = 6.0


@dataclass(frozen=True)
class RoleModel:
    """Retta di un ruolo. `coefficients` vuota significa: usa `fallback`."""

    coefficients: tuple[float, ...]
    intercept: float
    fallback: float


def fit(history: pd.DataFrame, roles: pd.Series, target: pd.Series) -> dict[str, RoleModel]:
    """Taratura per ruolo, ai minimi quadrati.

    I tre argomenti sono paralleli: la riga i-esima di ciascuno descrive la
    stessa osservazione. Per questo il mascheramento avviene su array numpy
    posizionali, non su Series pandas allineate per etichetta d'indice: se
    history avesse un indice diverso da roles/target, un `&` fra Series si
    allineerebbe per indice e produrrebbe coefficienti sbagliati (o un
    fallback ingiustificato) in silenzio.
    """
    if not len(history) == len(roles) == len(target):
        raise ValueError("history, roles e target devono avere lo stesso numero di righe")

    models: dict[str, RoleModel] = {}
    features = list(CALIBRATION_FEATURES)

    # fillna("") prima di confrontare: roles puo' essere dtype nullable
    # "string" (cosi' la produce ingest.py), dove `Series == valore` su una
    # riga NA da' pd.NA anziche' False e fa esplodere `.any()`/`&` a valle.
    # La stringa vuota non eguaglia mai un ruolo vero, quindi la riga resta
    # semplicemente esclusa da ogni ruolo, come deve essere.
    role_values = roles.fillna("").astype(str).to_numpy()
    target_values = pd.to_numeric(target, errors="coerce").to_numpy(np.float64)
    feature_values = history[features].to_numpy(np.float64)
    has_target = ~np.isnan(target_values)
    has_features = ~np.isnan(feature_values).any(axis=1)

    for role in sorted(roles.dropna().unique()):
        in_role = (role_values == role) & has_target
        fallback = float(target_values[in_role].mean()) if in_role.any() else DEFAULT_VOTE

        usable = in_role & has_features
        if int(usable.sum()) < MIN_SAMPLES:
            models[role] = RoleModel((), 0.0, fallback)
            continue

        matrix = feature_values[usable]
        design = np.column_stack([matrix, np.ones(len(matrix))])
        solution, *_ = np.linalg.lstsq(design, target_values[usable], rcond=None)
        models[role] = RoleModel(tuple(solution[:-1]), float(solution[-1]), fallback)

    return models


def predict(models: dict[str, RoleModel], history: pd.DataFrame, roles: pd.Series) -> pd.Series:
    """Voto stimato per ogni riga. Mai NaN: dove manca la storia, la media del ruolo."""
    features = list(CALIBRATION_FEATURES)
    out = pd.Series(DEFAULT_VOTE, index=history.index, dtype="float64")

    # stesso motivo del fillna in fit: roles nullable "string" con una riga
    # NA farebbe esplodere il confronto piu' sotto.
    role_values = roles.fillna("").astype(str).to_numpy()

    for role, model in models.items():
        in_role = role_values == role
        if not in_role.any():
            continue
        out.loc[in_role] = model.fallback
        if not model.coefficients:
            continue

        usable = in_role & history[features].notna().all(axis=1).to_numpy()
        if not usable.any():
            continue
        matrix = history.loc[usable, features].to_numpy(np.float64)
        out.loc[usable] = matrix @ np.array(model.coefficients) + model.intercept

    return out
