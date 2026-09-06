"""Non un numero solo, ma la forbice: quanto puo' andare bene e quanto male.

Due giocatori entrambi a 6,3 fantapunti attesi sono decisioni completamente
diverse se uno prende 6 quasi sempre e l'altro alterna 4,5 e 9,5. La media
non distingue i due casi, e sono esattamente i due casi in cui un
fantallenatore sceglie in modo opposto: chi insegue vuole la coda alta, chi
amministra un vantaggio vuole non prendere il 4.

La dispersione non e' un dettaglio del voto. Misurata sull'archivio, per
chi e' sceso in campo:

    ruolo   dev.st. del voto   dev.st. del fantavoto
        P              0.458                   1.881
        D              0.498                   1.021
        C              0.482                   1.312
        A              0.634                   1.951

Per un attaccante il fantavoto oscilla tre volte il voto: quasi tutta la
varianza viene dai gol, non dal giudizio del redattore. **Azzeccare la
probabilita' di gol conta piu' che raffinare la stima del voto**, ed e' la
ragione per cui gli interventi sulle quote vengono prima di quelli sul voto.

Simulazione e non formula chiusa: la somma di un voto continuo e di dieci
conteggi pesati, il tutto mescolato con la probabilita' di non giocare
affatto, non ha una distribuzione con un nome. Campionare costa niente e
gestisce da solo la miscela.
"""

import numpy as np
import pandas as pd

from fantasquama.scoring import EVENTS, Rules

# **Il voto e gli eventi non sono indipendenti.** Chi segna non prende solo
# i tre punti di bonus: prende anche un voto piu' alto di un punto pieno.
# Misurato sull'archivio, voto medio di chi ha fatto l'evento contro chi no:
#
#     gol  +1,07     assist  +0,72     ammonizione  -0,32
#     porta inviolata (portieri)  +0,50
#
# Campionarli separatamente rende la coda alta troppo sottile: la giornata
# da 9 non e' «voto alto OPPURE gol», e' «gol E percio' voto alto». Qui il
# voto campionato si sposta con gli eventi campionati, coi coefficienti di
# una regressione del voto sugli eventi realmente avvenuti.
#
# Lo scostamento e' CENTRATO sul valore atteso dell'evento, quindi la media
# non cambia -- resta quella del calibratore -- e cambia solo la forma.
VOTE_COEFFICIENTS: dict[str, float] = {
    "gf": 1.05, "rf": 1.06, "rs": -0.50, "ass": 0.68,
    "amm": -0.31, "esp": -0.85, "au": -0.44,
    "gs": -0.07, "rp": 0.85, "cs": 0.39,
}

# Quanto resta di imprevedibile nel voto una volta noti gli eventi. E' la
# dispersione RESIDUA di quella regressione, non quella grezza: usare la
# grezza conterebbe due volte la parte gia' spiegata dagli eventi.
#
#     ruolo   grezza   residua
#         P    0.458     0.359
#         D    0.498     0.404
#         C    0.482     0.321
#         A    0.634     0.344
#
# Per un attaccante gli eventi spiegano quasi meta' della varianza del voto.
VOTE_SD: dict[str, float] = {"P": 0.359, "D": 0.404, "C": 0.321, "A": 0.344}
DEFAULT_VOTE_SD: float = 0.36

# Il voto sta in questo intervallo: una normale non lo sa e va tagliata.
VOTE_MIN, VOTE_MAX = 1.0, 10.0

# `cs` e' una probabilita' (la porta o e' inviolata o non lo e'), tutti gli
# altri sono conteggi che possono capitare piu' volte nella stessa partita.
BERNOULLI_EVENTS: tuple[str, ...] = ("cs",)

# Le soglie che separano una giornata buona da una da dimenticare. Sette e'
# il fantavoto che di solito fa vincere la giornata a chi lo prende; cinque e'
# quello che la fa perdere.
HIGH: float = 7.0
LOW: float = 5.0

SAMPLES: int = 5000


def distribution(
    probabilities: pd.DataFrame,
    roles: pd.Series,
    votes: pd.Series,
    rules: Rules,
    samples: int = SAMPLES,
    seed: int = 20260905,
    high: float = HIGH,
    low: float = LOW,
) -> pd.DataFrame:
    """La distribuzione del fantavoto di ogni riga, per campionamento.

    Ritorna media, probabilita' di superare `high`, probabilita' di restare
    sotto `low`, e i quantili 10/50/90.

    **Costa memoria proporzionale alle righe per i campioni**: con 76.000
    righe e 5.000 campioni sarebbero tre miliardi di numeri. Va chiamata
    sulle sole righe che servono davvero -- la giornata da giocare -- e non
    sull'archivio intero. Per questo non e' uno stadio della pipeline ma una
    funzione a parte, che il chiamante invoca su cio' che gli interessa.

    Il seme e' fisso: due esecuzioni sullo stesso ingresso devono dare lo
    stesso numero, altrimenti la forbice mostrata all'utente cambierebbe a
    ogni aggiornamento senza che sia cambiato niente.
    """
    if not len(probabilities) == len(roles) == len(votes):
        raise ValueError("probabilita', ruoli e voti devono avere lo stesso numero di righe")
    if probabilities.empty:
        return pd.DataFrame(
            columns=["expected", "p_high", "p_low", "q10", "q50", "q90"]
        )

    rng = np.random.default_rng(seed)
    n = len(probabilities)
    role_values = roles.fillna("").astype(str).to_numpy()

    # Il voto: una normale attorno alla stima, con la dispersione del ruolo.
    sigma = np.array([VOTE_SD.get(r, DEFAULT_VOTE_SD) for r in role_values])
    centro = pd.to_numeric(votes, errors="coerce").fillna(6.0).to_numpy(np.float64)
    fantavoto = np.clip(
        rng.normal(centro[:, None], sigma[:, None], size=(n, samples)),
        VOTE_MIN, VOTE_MAX,
    )

    # Gli eventi, uno alla volta dentro l'accumulatore: tenerli tutti insieme
    # moltiplicherebbe per dieci la memoria senza servire a niente.
    #
    # Ogni evento entra due volte: col peso del regolamento, e con il suo
    # effetto sul voto. Il secondo e' centrato sul valore atteso, quindi
    # sposta la forma della distribuzione e non la sua media.
    # I gol subiti si campionano per primi perche' la porta inviolata non e'
    # un evento a se': **e' la stessa cosa detta in un altro modo**, cioe'
    # `gs == 0`. Campionandoli separatamente si producono giornate impossibili
    # -- porta inviolata con due gol subiti -- e la distribuzione del portiere
    # ne esce schiacciata verso il centro proprio dove serve distinguere.
    gol_subiti = rng.poisson(
        np.maximum(
            pd.to_numeric(probabilities["gs"], errors="coerce").fillna(0.0).to_numpy(np.float64), 0.0
        )[:, None],
        size=(n, samples),
    ).astype(np.float64)
    porta_inviolata = (gol_subiti == 0).astype(np.float64)
    # solo il portiere prende bonus e malus da questi due
    portiere = (role_values == "P").astype(np.float64)[:, None]
    gol_subiti *= portiere
    porta_inviolata *= portiere

    campionati: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "gs": (gol_subiti, pd.to_numeric(probabilities["gs"], errors="coerce")
               .fillna(0.0).to_numpy(np.float64)[:, None] * portiere),
        "cs": (porta_inviolata, porta_inviolata.mean(axis=1, keepdims=True)),
    }

    for evento in EVENTS:
        peso = getattr(rules, evento)
        effetto = VOTE_COEFFICIENTS.get(evento, 0.0)
        if peso == 0.0 and effetto == 0.0:
            continue
        if evento in campionati:
            campioni, atteso = campionati[evento]
        else:
            atteso = pd.to_numeric(probabilities[evento], errors="coerce").fillna(0.0)
            atteso = np.maximum(atteso.to_numpy(np.float64), 0.0)[:, None]
            if evento in BERNOULLI_EVENTS:
                atteso = np.clip(atteso, 0.0, 1.0)
                campioni = (rng.random((n, samples)) < atteso).astype(np.float64)
            else:
                campioni = rng.poisson(atteso, size=(n, samples)).astype(np.float64)
        fantavoto += peso * campioni
        if effetto:
            fantavoto += effetto * (campioni - atteso)

    # Chi non scende in campo vale il sostituto, non il proprio fantavoto:
    # e' la stessa miscela di `expected_points`, campionata invece che mediata.
    gioca = rng.random((n, samples)) < pd.to_numeric(
        probabilities["p_vote"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)[:, None]
    fantavoto = np.where(gioca, fantavoto, rules.sv)

    quantili = np.quantile(fantavoto, [0.10, 0.50, 0.90], axis=1)
    return pd.DataFrame(
        {
            "expected": fantavoto.mean(axis=1),
            "p_high": (fantavoto >= high).mean(axis=1),
            "p_low": (fantavoto <= low).mean(axis=1),
            "q10": quantili[0],
            "q50": quantili[1],
            "q90": quantili[2],
        },
        index=probabilities.index,
    )
