"""Il modello consiglia meglio della scelta ovvia?

Per ogni giornata si campionano coppie di giocatori dello stesso ruolo. Per
ogni coppia il modello dice chi rendera' di piu', e si confronta con l'esito
reale. Le baseline fanno la stessa cosa guardando solo chi gioca di piu'.

Il campione e' bilanciato per ruolo (stesso numero di coppie per ruolo), ma la
riga "tutti" li ricompone con il peso delle coppie che esistono davvero:
altrimenti i portieri, che sono il 5,8% delle coppie possibili, peserebbero
per il 22,8%.

Se il modello non batte la baseline, il progetto FantaSquama non ha ragione di
esistere, e lo abbiamo scoperto senza scrivere una riga di Swift.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fantasquama import calibrate
from fantasquama.estimate import (
    apply_match_context,
    blended_history,
    event_probabilities,
    previous_season,
    score_probabilities,
)
from fantasquama.fixtures import attach, fit_difficulty, load_fixtures, team_factors
from fantasquama.features import rolling_history
from fantasquama.ingest import load_archive
from fantasquama.scoring import EVENTS, Rules, fantavoto

PAIRS_PER_GAMEWEEK: int = 2000
FIRST_GAMEWEEK: int = 6
MARGIN_REQUIRED: float = 0.03
ROLES: tuple[str, ...] = ("P", "D", "C", "A")

# Sotto questo numero di coppie una sola risposta diversa sposta l'accuratezza
# di mezzo punto percentuale, un sesto della soglia del cancello: il rapporto
# non e' abbastanza fine per decidere il progetto.
MIN_PAIRS: int = 200

BASELINES: dict[str, str] = {
    "baseline_apps": "presenze cumulate da inizio stagione",
    "baseline_last5": "presenze nelle ultime cinque giornate",
}

# La spec 8 ordina la baseline per MINUTI giocati nelle ultime cinque presenze.
# L'archivio storico dei voti non contiene i minuti: nessuna delle due baseline
# qui sotto e' quella della spec, entrambe sostituiscono i minuti con le
# presenze. Il rapporto le mostra tutte e due e il verdetto usa la piu' forte,
# perche' la differenza fra "recente" e "cumulata" non e' un dettaglio: una
# baseline piu' debole della vera renderebbe il cancello troppo facile, cioe'
# rischierebbe un falso positivo -- costruire un'app che non funziona.
SUBSTITUTION_NOTE: str = (
    "Nota: la spec 8 ordina la baseline per minuti giocati nelle ultime cinque\n"
    "presenze. L'archivio storico dei voti non contiene i minuti, quindi qui\n"
    "sono sostituiti dalle presenze. Le due baseline sono entrambe surrogati:\n"
    "  baseline_apps   = presenze cumulate da inizio stagione (satura)\n"
    "  baseline_last5  = presenze nelle ultime cinque giornate (recente)\n"
    "Il verdetto applica il criterio alla piu' forte delle due."
)


def sample_pairs(frame: pd.DataFrame, rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Campiona `n` coppie di giocatori distinti dello stesso ruolo.

    Un ruolo con meno di due giocatori non ha nessuna coppia possibile e
    viene saltato. Il raggruppamento per ruolo lavora su un array numpy, non
    sulla colonna pandas: `role` puo' avere dtype nullable "string" (come lo
    produce ingest.py), dove un valore mancante renderebbe ambiguo il
    confronto `frame["role"] == valore`.
    """
    role_values = frame["role"].fillna("").astype(str).to_numpy()
    parts = []
    for role in pd.unique(role_values):
        pool = frame.iloc[np.flatnonzero(role_values == role)].reset_index(drop=True)
        if len(pool) < 2:
            continue
        size = min(n, len(pool) * (len(pool) - 1) // 2)
        left = rng.integers(0, len(pool), size=size * 2)
        right = rng.integers(0, len(pool), size=size * 2)
        keep = left != right
        # la stessa maschera su entrambi i lati: restano sempre della stessa
        # lunghezza, quindi il concat per colonna piu' sotto e' sempre valido
        left, right = left[keep][:size], right[keep][:size]

        a = pool.iloc[left].reset_index(drop=True).add_suffix("_a")
        b = pool.iloc[right].reset_index(drop=True).add_suffix("_b")
        parts.append(pd.concat([a, b], axis=1))

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def pairwise_accuracy(pairs: pd.DataFrame, score_column: str) -> float:
    """Quante volte chi era indicato come migliore ha davvero reso di piu'.

    Le coppie con esito reale identico sono escluse: non c'e' una risposta
    giusta. I pareggi del punteggio valgono mezzo punto, come una monetina.
    """
    decided = pairs[pairs["actual_a"] != pairs["actual_b"]]
    if decided.empty:
        return float("nan")

    predicted = np.sign(decided[f"{score_column}_a"] - decided[f"{score_column}_b"])
    truth = np.sign(decided["actual_a"] - decided["actual_b"])
    return float(np.where(predicted == 0, 0.5, (predicted == truth).astype(float)).mean())


def _require_seasons(
    mask: np.ndarray, archive: pd.DataFrame, requested: list[str], what: str
) -> None:
    """Ferma il backtest se una lista di stagioni non seleziona nessuna riga.

    Un refuso in --train svuota la maschera: `calibrate.fit` non tara niente,
    `predict` ritorna il voto di default su ogni riga e il backtest arriva in
    fondo con un rapporto indistinguibile da uno vero. Lo stesso refuso in
    --test da' un rapporto tutto NaN e un "il progetto si ferma qui" che e' un
    falso negativo sul cancello del progetto. In entrambi i casi il modo di
    fallire e' silenzioso, quindi il controllo dev'essere esplicito.
    """
    if mask.any():
        return
    presenti = sorted(archive["season"].dropna().unique())
    raise ValueError(
        f"nessuna riga per le stagioni di {what} {sorted(requested)}; "
        f"nell'archivio ci sono: {presenti}"
    )


def all_pairs_weights(table: pd.DataFrame) -> dict[str, float]:
    """Quante coppie dello stesso ruolo esistono davvero, giornata per giornata.

    `sample_pairs` ne estrae PAIRS_PER_GAMEWEEK per ruolo a prescindere dalla
    dimensione del ruolo, quindi il campione e' bilanciato per ruolo, non
    proporzionale. Con le rose reali della Serie A (P 60, D 160, C 160, A 80)
    i portieri sono il 5,8% delle coppie possibili ma il 22,8% di quelle
    campionate: quasi quattro volte troppo, proprio nel ruolo dove il modello
    ha il vantaggio strutturale piu' grande (`rp` e `gs` esistono solo li').

    Questi conteggi sono i pesi con cui la riga "tutti" ricompone i quattro
    ruoli. Ho scelto di pesare invece di riallocare il campionamento perche'
    riallocarlo avrebbe ridotto le coppie dei portieri di quattro volte,
    peggiorando la stima proprio del ruolo su cui il criterio della spec 8
    richiede comunque un miglioramento: pesare corregge la riga "tutti"
    senza togliere precisione a nessuna riga di ruolo.
    """
    keyed = table.assign(role_key=table["role"].fillna("").astype(str))
    sizes = keyed.groupby(["season", "gameweek", "role_key"], sort=False).size()
    combinations = sizes * (sizes - 1) // 2
    totals = combinations.groupby(level="role_key").sum()
    return {role: float(totals.get(role, 0.0)) for role in ROLES}


def _weighted(rows: list[dict], column: str, weights: dict[str, float]) -> float:
    """Media delle accuratezze di ruolo, pesata sulle coppie realmente esistenti.

    I ruoli senza coppie decise hanno accuratezza NaN e restano fuori dalla
    media: non contribuiscono ne' al numeratore ne' al denominatore.
    """
    usable = [
        (weights.get(row["role"], 0.0), row[column])
        for row in rows
        if pd.notna(row[column]) and weights.get(row["role"], 0.0) > 0.0
    ]
    total = sum(weight for weight, _ in usable)
    if total == 0.0:
        return float("nan")
    return sum(weight * value for weight, value in usable) / total


def _scope(table: pd.DataFrame, dropped: int) -> str:
    """Che cosa e' stato davvero misurato, in chiaro sotto il rapporto.

    Il criterio della spec 8 parla dell'"intera stagione", mentre il backtest
    parte dalla giornata FIRST_GAMEWEEK: prima non c'e' abbastanza storia
    perche' una stima significhi qualcosa. La differenza va detta, non
    lasciata dedurre dal codice.

    `dropped` sono le righe rimosse da `dropna`. Oggi e' zero; se un giorno
    non lo fosse, il campione del cancello sarebbe auto-selezionato e il
    numero stampato sarebbe credibile e sbagliato: per questo si stampa
    sempre, anche quando vale zero.
    """
    if table.empty:
        return "Perimetro: nessuna riga nel campione."
    seasons = ", ".join(sorted(table["season"].dropna().unique()))
    gameweeks = table["gameweek"].to_numpy()
    return (
        f"Perimetro: stagioni {seasons}, giornate da {int(gameweeks.min())} a "
        f"{int(gameweeks.max())} (le prime {FIRST_GAMEWEEK - 1} sono escluse: "
        f"senza storia pregressa la stima non significa niente), "
        f"{len(table)} righe, {table['player_id'].nunique()} giocatori, "
        f"{dropped} righe scartate per valori mancanti."
    )


def run(
    archive_root: Path,
    rules: Rules,
    train_seasons: list[str],
    test_seasons: list[str],
) -> pd.DataFrame:
    """Esegue il backtest e ritorna un rapporto per ruolo, piu' la riga 'tutti'.

    Il calibratore si tara SOLO sulle stagioni di allenamento; le stagioni di
    verifica entrano solo in `calibrate.predict`, mai in `calibrate.fit`. Le
    due liste non possono sovrapporsi: una stagione usata per tarare il
    modello e poi per giudicarlo renderebbe il risultato senza senso, nel modo
    piu' difficile da notare, quindi il controllo avviene prima di leggere
    l'archivio.

    La stessa maschera va a `expected_fantapoints`: anche i prior dello
    shrinkage si calcolano sulle sole stagioni di taratura. Il confine fra
    cio' che il modello ha visto e cio' che deve prevedere e' quindi uno solo
    in tutto il backtest.
    """
    overlap = set(train_seasons) & set(test_seasons)
    if overlap:
        raise ValueError(
            f"stagioni di taratura e di verifica sovrapposte: {sorted(overlap)}"
        )

    archive = load_archive(archive_root)
    history = rolling_history(archive)

    # il fantavoto realmente ottenuto: chi non prende voto vale sv
    played = archive["played"].to_numpy()
    actual = np.where(
        played,
        [
            fantavoto(float(v) if pd.notna(v) else 0.0, row, rules)
            for v, row in zip(archive["voto"], archive[list(EVENTS)].to_dict("records"))
        ],
        rules.sv,
    )

    # mascheramento posizionale: rolling_history garantisce a `history` lo
    # stesso ordine e lo stesso indice di `archive`, quindi un array numpy
    # booleano su .iloc tiene i tre argomenti di fit allineati riga per riga.
    train_mask = archive["season"].isin(train_seasons).to_numpy()
    _require_seasons(train_mask, archive, train_seasons, "taratura")

    # il calibratore legge le medie dritte, quindi gliele si passa gia'
    # attenuate verso la stagione scorsa: altrimenti a inizio stagione legge
    # zeri e stima lo stesso voto per tutti
    previous = previous_season(archive)
    blended = blended_history(history, previous)

    models = calibrate.fit(
        blended.iloc[train_mask],
        archive["role"].iloc[train_mask],
        archive["voto"].iloc[train_mask],
    )

    votes = calibrate.predict(models, blended, archive["role"])
    # stessa maschera del calibratore: i prior dello shrinkage si calcolano
    # sulle sole stagioni di taratura. Sull'intero archivio, la stima di una
    # giornata leggerebbe le giornate successive della stagione da giudicare.
    # il prior di ogni riga parte da come quel giocatore ha giocato la
    # stagione PRIMA: e' cio' che rende leggibile l'inizio di stagione, quando
    # di quella in corso non si sa ancora niente. Non serve mascherarlo -- la
    # stagione precedente e' passato per definizione.
    probabilities = event_probabilities(history, archive["role"], train_mask, previous)

    # La difficolta' della partita entra fra le probabilita' e il punteggio.
    # Se il calendario non c'e' il modello resta quello di prima: senza
    # avversario non c'e' correzione da applicare, e fermarsi renderebbe il
    # backtest ineseguibile per chi ha solo l'archivio dei voti.
    fixtures_root = archive_root / "fixtures"
    con_calendario = fixtures_root.exists()
    if con_calendario:
        fixtures = load_fixtures(fixtures_root)
        context = attach(archive, fixtures)
        advantage = (context["p_win"] - context["p_lose"]).to_numpy(np.float64)
        market_attack, market_defense = fit_difficulty(fixtures, train_seasons).factors(advantage)
        squad_attack, squad_defense = team_factors(context)
        has_market = np.isfinite(advantage)
        attack = market_attack * np.where(has_market, squad_attack ** 0.20, squad_attack)
        defense = market_defense * np.where(has_market, squad_defense ** 0.20, squad_defense)
        probabilities = apply_match_context(probabilities, attack, defense)

    scores = score_probabilities(probabilities, archive["role"], votes, rules)

    table = pd.DataFrame({
        "season": archive["season"],
        "gameweek": archive["gameweek"],
        "player_id": archive["player_id"],
        "role": archive["role"],
        "score": scores,
        "baseline_apps": history["apps_before"],
        "baseline_last5": history["apps_last5"],
        "actual": actual,
    })
    test_mask = archive["season"].isin(test_seasons).to_numpy()
    _require_seasons(test_mask, archive, test_seasons, "verifica")

    table = table[test_mask & (table["gameweek"] >= FIRST_GAMEWEEK).to_numpy()]

    # dropna che conta: righe che spariscono in silenzio dal campione del
    # cancello sono un campione auto-selezionato, cioe' un numero credibile e
    # sbagliato. Vedi la nota nel docstring di `scope`.
    needed = ["score", "baseline_apps", "baseline_last5", "actual"]
    dropped = int(len(table) - len(table.dropna(subset=needed)))
    table = table.dropna(subset=needed)

    rng = np.random.default_rng(20260823)
    groups = [
        sample_pairs(group, rng, PAIRS_PER_GAMEWEEK)
        for _, group in table.groupby(["season", "gameweek"], sort=True)
    ]
    groups = [g for g in groups if not g.empty]
    if groups:
        pairs = pd.concat(groups, ignore_index=True)
    else:
        # nessuna coppia da nessuna giornata: un frame vuoto ma con le
        # colonne attese, cosi' il resto della funzione non deve
        # distinguere questo caso da un risultato normale
        empty_columns = [f"{c}_a" for c in table.columns] + [f"{c}_b" for c in table.columns]
        pairs = pd.DataFrame(columns=empty_columns)

    # confronto per ruolo su array numpy, non sulla colonna pandas: stesso
    # motivo di sample_pairs, "role_a" ha dtype nullable "string"
    role_a_values = pairs["role_a"].fillna("").astype(str).to_numpy()

    rows = []
    for role in ROLES:
        subset = pairs.iloc[np.flatnonzero(role_a_values == role)]
        rows.append({
            "role": role,
            "model": pairwise_accuracy(subset, "score"),
            "baseline_apps": pairwise_accuracy(subset, "baseline_apps"),
            "baseline_last5": pairwise_accuracy(subset, "baseline_last5"),
            "pairs": len(subset),
        })
    # la riga "tutti" ricompone i ruoli con il peso delle coppie che
    # esistono davvero, non con quello del campione: vedi all_pairs_weights
    weights = all_pairs_weights(table)
    rows.append({
        "role": "tutti",
        "model": _weighted(rows, "model", weights),
        "baseline_apps": _weighted(rows, "baseline_apps", weights),
        "baseline_last5": _weighted(rows, "baseline_last5", weights),
        "pairs": len(pairs),
    })

    report = pd.DataFrame(rows)
    report.attrs["scope"] = _scope(table, dropped)
    return report


def verdict(report: pd.DataFrame) -> tuple[bool, str]:
    """Applica il criterio della spec 8 e spiega l'esito.

    Il criterio si applica alla piu' forte delle due baseline, quella con
    l'accuratezza complessiva piu' alta: batterne una debole non dimostra
    niente, e il rischio da evitare qui e' il falso positivo -- passare il
    cancello e costruire un'app che non aggiunge valore. La stessa baseline
    vale anche per il controllo ruolo per ruolo, cosi' l'esito dipende da un
    solo termine di confronto, dichiarato nel testo.

    Il numero di coppie non e' decorativo: con un campione minuscolo il
    margine e' rumore, e un rapporto con una coppia per ruolo passerebbe con
    un trionfale +100%.
    """
    overall = report[report["role"] == "tutti"].iloc[0]

    scarse = [
        f"{row['role']} ({int(row['pairs'])})"
        for _, row in report.iterrows()
        if int(row["pairs"]) < MIN_PAIRS
    ]
    if scarse:
        return False, (
            f"Campione troppo piccolo per decidere: servono almeno {MIN_PAIRS} "
            f"coppie per riga, mancano su {', '.join(scarse)}."
        )

    # la baseline piu' forte e' quella piu' difficile da battere
    column = max(BASELINES, key=lambda c: (overall[c] if pd.notna(overall[c]) else -1.0))
    named = f"{column} ({BASELINES[column]})"
    margin = overall["model"] - overall[column]

    weak = [
        row["role"]
        for _, row in report[report["role"] != "tutti"].iterrows()
        if not (row["model"] > row[column])
    ]

    if not (margin >= MARGIN_REQUIRED):
        return False, (
            f"Rispetto alla baseline piu' forte, {named}, il modello guadagna "
            f"{margin:+.1%}: sotto la soglia richiesta del {MARGIN_REQUIRED:.0%}."
        )
    if weak:
        return False, (
            f"Rispetto alla baseline piu' forte, {named}, il margine "
            f"complessivo e' {margin:+.1%}, ma non c'e' miglioramento su "
            f"questi ruoli: {', '.join(weak)}."
        )
    return True, (
        f"Rispetto alla baseline piu' forte, {named}, il modello guadagna "
        f"{margin:+.1%} su tutti i ruoli."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest di FantaSquama")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--train", nargs="+", required=True, help="stagioni di taratura")
    parser.add_argument("--test", nargs="+", required=True, help="stagioni di verifica")
    args = parser.parse_args()

    report = run(args.data, Rules(), args.train, args.test)
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(report.attrs["scope"])
    print()
    print(SUBSTITUTION_NOTE)
    print()

    passed, explanation = verdict(report)
    print(explanation)
    print("ESITO:", "il progetto ha senso, si procede" if passed else "il progetto si ferma qui")


if __name__ == "__main__":
    main()
