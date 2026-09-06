"""Le domande che l'accuratezza media non risponde.

Una percentuale sola dice se il modello ordina meglio della baseline, e non
dice nient'altro: non se sa quando e' sicuro, non se le probabilita' che
stampa sono probabilita' vere, e non quanti fantapunti fa guadagnare a chi lo
segue. Sono tre cose diverse, e un modello puo' essere bravo in una e inutile
nelle altre.
"""

import numpy as np
import pandas as pd

# I quattro scaglioni di distanza fra i punteggi predetti. Un modello sano e'
# monotono su questi: piu' e' largo il divario che dichiara, piu' spesso ha
# ragione. Se non lo e', il problema non e' quanto segnale ha ma quanto si
# fida di se stesso -- ed e' un difetto che l'accuratezza media non mostra.
CONFIDENCE_EDGES: tuple[float, ...] = (0.0, 0.3, 0.8, 1.5, np.inf)


def confidence_buckets(pairs: pd.DataFrame, score: str = "score") -> pd.DataFrame:
    """Accuratezza per ampiezza del divario predetto.

    Le coppie con esito reale identico restano fuori, come in
    `pairwise_accuracy`: non c'e' una risposta giusta da indovinare.
    """
    if pairs.empty:
        return pd.DataFrame(columns=["fascia", "coppie", "divario", "accuratezza"])

    decise = pairs[pairs["actual_a"] != pairs["actual_b"]]
    divario = (decise[f"{score}_a"] - decise[f"{score}_b"]).abs().to_numpy()
    predetto = np.sign(decise[f"{score}_a"] - decise[f"{score}_b"])
    vero = np.sign(decise["actual_a"] - decise["actual_b"])
    giusto = np.where(predetto == 0, 0.5, (predetto == vero).astype(float))

    righe = []
    for basso, alto in zip(CONFIDENCE_EDGES, CONFIDENCE_EDGES[1:]):
        dentro = (divario >= basso) & (divario < alto)
        if not dentro.any():
            continue
        etichetta = f"{basso:.1f}-{alto:.1f}" if np.isfinite(alto) else f"oltre {basso:.1f}"
        righe.append({
            "fascia": etichetta,
            "coppie": int(dentro.sum()),
            "divario": float(divario[dentro].mean()),
            "accuratezza": float(giusto[dentro].mean()),
        })
    return pd.DataFrame(righe)


def is_monotone(buckets: pd.DataFrame) -> bool:
    """L'accuratezza cresce insieme al divario dichiarato?"""
    valori = buckets["accuratezza"].to_numpy()
    return bool(len(valori) < 2 or np.all(np.diff(valori) >= -0.005))


def top_n_value(table: pd.DataFrame, columns: list[str], n: int = 10) -> pd.DataFrame:
    """Quanto rende davvero seguire il consiglio, invece di quanto e' accurato.

    Per ogni giornata e ogni ruolo si prendono i primi `n` secondo ciascun
    criterio e si media il fantavoto che hanno **davvero** ottenuto. E' la
    metrica che corrisponde all'uso reale dell'app: nessuno schiera una
    coppia, si schiera una formazione.

    Un modello puo' vincere sull'accuratezza a coppie e perdere qui, se
    indovina tanti confronti piccoli e sbaglia i pochi che spostano punti.
    """
    righe = []
    for role, per_ruolo in table.groupby("role", sort=True, observed=True):
        medie: dict[str, list[float]] = {colonna: [] for colonna in columns}
        for _, giornata in per_ruolo.groupby(["season", "gameweek"], sort=True):
            if len(giornata) < n:
                continue
            for colonna in columns:
                scelti = giornata.nlargest(n, colonna)
                medie[colonna].append(float(scelti["actual"].mean()))
        if not any(medie.values()):
            continue
        righe.append({
            "role": str(role),
            "giornate": len(next(iter(medie.values()))),
            **{colonna: float(np.mean(valori)) for colonna, valori in medie.items()},
        })
    return pd.DataFrame(righe)


def brier(previste: np.ndarray, avvenute: np.ndarray) -> float:
    """Scarto quadratico medio fra probabilita' dichiarata ed esito.

    Piu' basso e' meglio. Il termine di paragone che conta non e' zero ma la
    costante che predice sempre la frequenza media: una probabilita' che non
    la batte non sta dicendo niente di quella riga in particolare.
    """
    usabili = np.isfinite(previste) & np.isfinite(avvenute)
    if not usabili.any():
        return float("nan")
    return float(((previste[usabili] - avvenute[usabili]) ** 2).mean())


# Gli eventi che il modello stima come CONTEGGIO atteso (gol per presenza,
# assist per presenza...) e non come probabilita'. `cs` non e' fra questi: la
# porta inviolata o c'e' o non c'e', e il modello ne stima direttamente la
# probabilita'.
COUNT_EVENTS: tuple[str, ...] = ("gf", "rf", "rs", "rp", "gs", "au", "amm", "esp", "ass")


def occurrence_probability(expected: np.ndarray) -> np.ndarray:
    """Da conteggio atteso a probabilita' che l'evento capiti almeno una volta.

    Serve perche' le due cose non sono la stessa e confrontarle direttamente
    da' un risultato falso: un attaccante con 0,25 gol attesi non ha il 25%
    di probabilita' di segnare, ne ha il 22,1%, perche' una parte di quei gol
    sta nelle giornate in cui ne fa due. Il conteggio atteso e' sempre >= la
    probabilita', quindi misurare il Brier sul conteggio grezzo fa sembrare
    ogni modello mal tarato per eccesso -- un difetto dello strumento, non
    del modello.

    Sotto Poisson, `P(almeno uno) = 1 - e^-lambda`.
    """
    return 1.0 - np.exp(-np.maximum(np.asarray(expected, dtype=np.float64), 0.0))


def event_calibration(
    probabilities: pd.DataFrame, archive: pd.DataFrame, mask: np.ndarray,
    events: tuple[str, ...] = ("gf", "ass", "amm", "cs"),
) -> pd.DataFrame:
    """Brier di ogni evento, contro la costante che predice la media.

    Solo le righe in cui il giocatore ha davvero giocato: la probabilita' di
    segnare di chi non e' sceso in campo non e' un esito osservabile, e
    contarla misurerebbe `p_vote` una seconda volta invece dell'evento.

    `cs` si guarda sui soli portieri -- e' l'unico ruolo a cui frutta bonus.
    """
    giocate = mask & archive["played"].to_numpy(bool)
    portieri = archive["role"].fillna("").astype(str).to_numpy() == "P"

    righe = []
    for evento in events:
        selezione = giocate & portieri if evento == "cs" else giocate
        if not selezione.any():
            continue
        previste = pd.to_numeric(probabilities[evento], errors="coerce").to_numpy()[selezione]
        if evento in COUNT_EVENTS:
            previste = occurrence_probability(previste)
        if evento == "cs":
            avvenute = (pd.to_numeric(archive["gs"], errors="coerce").to_numpy()[selezione] == 0)
        else:
            avvenute = pd.to_numeric(archive[evento], errors="coerce").to_numpy()[selezione] > 0
        avvenute = avvenute.astype(float)

        modello = brier(previste, avvenute)
        costante = brier(np.full(len(avvenute), avvenute.mean()), avvenute)
        righe.append({
            "evento": evento,
            "righe": int(selezione.sum()),
            "frequenza": float(avvenute.mean()),
            "brier": modello,
            "brier_costante": costante,
            "meglio_della_costante": bool(modello < costante),
        })
    return pd.DataFrame(righe)
