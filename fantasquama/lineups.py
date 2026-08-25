"""Probabili formazioni: chi gioca, chi sta fuori, chi batte i piazzati.

L'archivio dice come un giocatore ha reso quando ha giocato. Non dice se
domenica gioca. Fino a qui il modello lo deduceva dai minuti dell'anno
scorso -- che alla prima giornata e' tutto quello che c'e', ma resta una
deduzione: un portiere appena arrivato dall'estero non ha minuti in Serie A
e riceve la media del ruolo, anche quando e' il titolare designato.

Le probabili formazioni rispondono direttamente. Qui non c'e' niente da
prevedere: c'e' da sostituire una deduzione con un'informazione.

**La percentuale della fonte non e' la probabilita' di prendere voto.** Dice
quanto e' probabile che scenda in campo dal primo minuto, ed e' una cosa
diversa: un titolare sicuro puo' uscire al 20' senza voto, o restare fuori
per un problema dell'ultima ora. Il ponte fra le due e' `NAILED_STARTER_RATE`,
misurato sull'archivio.

Le formazioni valgono per la giornata che descrivono e per quella soltanto:
sono una previsione fatta prima che si giochi.

Nessuna rete: il file lo produce `fetch_lineups.py`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fantasquama import roster

SLOTS = ("titolare", "panchina", "indisponibile")

# P(prende voto alla giornata g | l'ha preso alla g-1 **e** alla g+1), per
# ruolo, misurata sulle tre stagioni dell'archivio (23.957 righe).
#
# La doppia condizione non e' un vezzo: seleziona chi in quel periodo era sano
# e in squadra, che e' esattamente cio' che la fonte dichiara quando scrive
# 100%. Condizionare solo sulla giornata prima darebbe 0,80, ma dentro
# quell'80% c'e' anche il rischio di infortunio -- che la fonte ha gia' tolto,
# elencando gli indisponibili a parte. Contarlo due volte punirebbe i titolari.
NAILED_STARTER_RATE: dict[str, float] = {"P": 0.961, "D": 0.874, "C": 0.875, "A": 0.873}

# P(prende voto | non l'ha preso la giornata prima, ma prima si), 19.120
# righe. E' il tetto di chi la fonte non nomina affatto: puo' entrare, non
# puo' essere trattato da titolare.
ROTATION_RATE: float = 0.285

# Rigori a favore per squadra per giornata (52 squadra-stagione, 5,68 a
# stagione) e quota presa dal primo, secondo e terzo rigorista dichiarato.
# I battitori per squadra sono in media 3,02: esattamente i tre che le fonti
# elencano, il che e' anche una conferma che l'elenco e' completo.
TEAM_PENALTIES_PER_GAMEWEEK: float = 0.1508
TAKER_SHARE: tuple[float, float, float] = (0.600, 0.240, 0.098)
RESIDUAL_SHARE: float = 0.044
PENALTY_CONVERSION: float = 0.771   # 263 segnati su 341 calciati

# Sotto questa quota di squadre coperte il sito ha cambiato pagina o e' fuori
# stagione, e proseguire vorrebbe dire mandare in campo mezza Serie A con la
# stima di ripiego senza che nulla lo dica.
MIN_TEAMS: int = 16

# Sotto questa quota di nomi agganciati al listone il parsing ha preso
# spazzatura, e le formazioni che ne escono non sono formazioni. Sopra, i nomi
# che restano fuori sono quasi sempre acquisti caduti fra la pubblicazione del
# listone e quella delle formazioni: reali, ma non in rosa per l'app.
MIN_ATTACH_RATE: float = 0.92


def load_probabili(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Il JSON di `fetch_lineups.py`, in due tabelle.

    La prima ha una riga per giocatore: squadra, slot, percentuale della
    fonte, e per gli indisponibili lo stato. La seconda ha una riga per
    squadra col modulo.

    Il referto che la fonte scrive accanto a ogni infortunio non entra: e'
    testo della loro redazione, e quello che cambia il consiglio e' che il
    giocatore non giochi -- per dirlo basta la parola.
    """
    data = json.loads(Path(path).read_text())
    giocatori: list[dict] = []
    squadre: list[dict] = []

    for partita in data.get("partite", []):
        for lato in ("casa", "ospite"):
            parte = partita.get(lato) or {}
            nome = parte.get("squadra")
            if not nome:
                continue
            squadre.append({
                "squadra": nome,
                "modulo": parte.get("formazione") or "",
                "data": partita.get("data") or "",
                # Nei JSON vecchi il campo non c'e': la stessa regola usata
                # dallo scraper mantiene l'interpretazione deterministica.
                "ufficiale": bool(parte.get("ufficiale")) or (
                    len(parte.get("titolari") or []) == 11
                    and all(float(r.get("titolarita_pct") or 0) == 100
                            for r in parte.get("titolari") or [])
                ),
                # L'elenco resta separato dai giocatori: sono coppie (o, se
                # la fonte lo fara', gruppi) e l'associazione e' informazione
                # utile alla schermata, non un nuovo ruolo del calciatore.
                "ballottaggi": parte.get("ballottaggi") or [],
            })
            for sezione, slot in (("titolari", "titolare"), ("panchina", "panchina")):
                for riga in parte.get(sezione, []):
                    giocatori.append({
                        "squadra": nome,
                        "giocatore": riga["giocatore"],
                        "slot": slot,
                        "titolarita": float(riga.get("titolarita_pct") or 0),
                        "stato": "",
                    })
            for riga in parte.get("indisponibili", []):
                giocatori.append({
                    "squadra": nome,
                    "giocatore": riga["giocatore"],
                    "slot": "indisponibile",
                    # chi non c'e' non gioca: la percentuale non serve
                    "titolarita": 0.0,
                    "stato": riga.get("stato") or "indisponibile",
                })

    if len({s["squadra"] for s in squadre}) < MIN_TEAMS:
        raise ValueError(
            f"solo {len({s['squadra'] for s in squadre})} squadre in {path} "
            f"(minimo {MIN_TEAMS}): il sito ha cambiato pagina, oppure e' fuori "
            "stagione. Rilanciare fetch_lineups.py e controllare l'output."
        )

    frame = pd.DataFrame(giocatori)
    undici = frame[frame["slot"] == "titolare"].groupby("squadra").size()
    sbagliate = undici[undici != 11]
    if not sbagliate.empty:
        raise ValueError(
            "una formazione ha undici nomi, non di piu' e non di meno:\n"
            f"{sbagliate.to_string()}"
        )
    return frame, pd.DataFrame(squadre).drop_duplicates("squadra").reset_index(drop=True)


def load_set_pieces(path: Path) -> pd.DataFrame:
    """Rigoristi e battitori da fermo, che la fonte delle formazioni non da'.

    Cambiano una volta a stagione, non una volta a settimana: per questo
    stanno in un file scritto a mano e non in uno scaricato.
    """
    table = pd.read_csv(path, comment="#", dtype={"squadra": "string", "giocatore": "string"})
    missing = {"squadra", "giocatore", "rigori", "fermo"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} deve avere le colonne {sorted(missing)}")
    for colonna in ("rigori", "fermo"):
        ordine = table[colonna].dropna()
        if not ordine.between(1, 3).all():
            raise ValueError(f"{colonna}: gli ordini validi sono 1, 2 e 3")
    return table.reset_index(drop=True)


def attach(lineups: pd.DataFrame, rosa: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggiunge a `lineups` il `listone_id` di ciascun nome.

    `rosa` e' l'uscita di `roster.match`. Il confronto e' lo stesso degli
    agganci fra listone e archivio, ma ristretto alla squadra, che qui e'
    nota: un cognome ambiguo in Serie A non lo e' quasi mai dentro una rosa.

    Ritorna il frame agganciato e l'elenco dei nomi lasciati indietro. Sotto
    `MIN_ATTACH_RATE` si ferma: vedi il commento sulla soglia.
    """
    per_squadra: dict[str, list[tuple]] = {}
    for row in rosa.itertuples():
        per_squadra.setdefault(str(row.team), []).append(
            (roster.name_tokens(row.player_name), roster.name_initial(row.player_name),
             str(row.listone_id))
        )

    found: list[str] = []
    persi: list[str] = []
    for row in lineups.itertuples():
        squadra = roster.TEAM_ALIASES.get(str(row.squadra), str(row.squadra))
        tokens = roster.name_tokens(row.giocatore)
        initial = roster.name_initial(row.giocatore)
        rosa_squadra = per_squadra.get(squadra, [])

        # Prima il nome identico, iniziale compresa. Il listone del Milan ha
        # «Terracciano» e «Terracciano F.», due persone: la regola larga li
        # confonde entrambi, quella esatta li separa senza ambiguita'.
        candidati = [
            listone_id for altri, altra_iniziale, listone_id in rosa_squadra
            if altri == tokens and altra_iniziale == initial
        ]
        if len(candidati) != 1 and tokens:
            candidati = [
                listone_id
                for altri, altra_iniziale, listone_id in rosa_squadra
                if (set(tokens) <= set(altri) or set(altri) <= set(tokens))
                and (not initial or not altra_iniziale or altra_iniziale.startswith(initial[:1]))
            ]

        if len(candidati) == 1:
            found.append(candidati[0])
        else:
            found.append("")
            persi.append(f"{row.squadra}: {row.giocatore}"
                         + (f" ({len(candidati)} omonimi)" if candidati else ""))

    # Un nome che resta fuori puo' voler dire due cose molto diverse, e la
    # differenza sta nel numero, non nel nome. Quattro nomi su cinquecento
    # sono acquisti caduti fra il listone e le formazioni: reali, ma per l'app
    # non esistono, quindi ignorarli non toglie niente. Meta' dei nomi fuori
    # significa che il parsing ha preso spazzatura, e allora ogni formazione
    # che ne esce e' finta -- ma girerebbe lo stesso, senza sintomi.
    rate = 1.0 - len(persi) / len(lineups) if len(lineups) else 0.0
    if rate < MIN_ATTACH_RATE:
        raise ValueError(
            f"solo il {rate:.0%} dei nomi si aggancia al listone (minimo "
            f"{MIN_ATTACH_RATE:.0%}): il parsing ha preso spazzatura, oppure "
            f"il listone e' di un'altra stagione.\n  " + "\n  ".join(persi[:20])
        )
    out = lineups.copy()
    out["listone_id"] = pd.array(found, dtype="string")
    return out, persi


def play_probability(slot: object, titolarita: float, role: str) -> float:
    """Da «quanto e' probabile che parta» a «quanto e' probabile che prenda voto».

    Chi e' indisponibile vale zero: non e' una stima bassa, e' un fatto.
    Chi la fonte non nomina affatto ricade sulla quota di rotazione.
    """
    if slot == "indisponibile":
        return 0.0
    base = NAILED_STARTER_RATE.get(role, NAILED_STARTER_RATE["C"])
    if slot in ("titolare", "panchina"):
        return base * max(0.0, min(titolarita, 100.0)) / 100.0
    return ROTATION_RATE


def penalty_attempts(rank: object) -> float:
    """Rigori calciati per giornata di squadra, dato il posto nell'elenco."""
    if pd.isna(rank):
        return TEAM_PENALTIES_PER_GAMEWEEK * RESIDUAL_SHARE
    posto = int(rank)
    if not 1 <= posto <= len(TAKER_SHARE):
        raise ValueError(f"ordine rigorista fuori scala: {rank}")
    return TEAM_PENALTIES_PER_GAMEWEEK * TAKER_SHARE[posto - 1]


def apply(
    probabilities: pd.DataFrame,
    roles: pd.Series,
    teams: pd.Series,
    slots: pd.Series,
    titolarita: pd.Series,
    penalty_rank: pd.Series,
) -> pd.DataFrame:
    """Sostituisce `p_vote` e i rigori con quello che dice la formazione.

    Tocca **solo le squadre che il file copre**: per le altre non c'e'
    informazione, e inventarne una sarebbe peggio che non averne.

    Gli eventi del modello sono per presenza, non per giornata: i rigori di
    squadra vanno quindi divisi per una probabilita' di essere in campo. Non
    per quella del singolo, pero': la quota del 60% e' misurata su rigoristi
    che erano titolari, e dividerla per la probabilita' di un panchinaro
    direbbe che uno che gioca un quarto delle partite, in quelle, batte tutti
    i rigori della squadra. Il divisore e' quindi la disponibilita' di un
    titolare del suo ruolo. Chi gioca meno ne prende meno lo stesso, perche' i
    fantapunti sono gia' moltiplicati per la sua probabilita' di giocare.
    """
    for colonna in (roles, teams, slots, titolarita, penalty_rank):
        if len(colonna) != len(probabilities):
            raise ValueError("tutte le colonne devono avere una riga per riga di probabilities")

    coperte = set(teams[slots.notna()].dropna().astype(str))
    if not coperte:
        return probabilities

    out = probabilities.copy()
    dentro = teams.astype(str).isin(coperte).to_numpy()
    role_values = roles.fillna("").astype(str).to_numpy()
    slot_values = slots.to_numpy(dtype=object)
    pct_values = pd.to_numeric(titolarita, errors="coerce").fillna(0.0).to_numpy()
    rank_values = penalty_rank.to_numpy(dtype=object)

    p_vote = out["p_vote"].to_numpy(np.float64).copy()
    rf = out["rf"].to_numpy(np.float64).copy()
    rs = out["rs"].to_numpy(np.float64).copy()

    for i in np.flatnonzero(dentro):
        atteso = play_probability(slot_values[i], pct_values[i], role_values[i])
        # chi la formazione non nomina non puo' valere piu' di un subentrante,
        # ma puo' valere meno: la sua storia resta l'informazione migliore
        p_vote[i] = atteso if slot_values[i] in SLOTS else min(p_vote[i], atteso)

        titolare = NAILED_STARTER_RATE.get(role_values[i], NAILED_STARTER_RATE["C"])
        calciati = penalty_attempts(rank_values[i]) / titolare
        if pd.isna(rank_values[i]):
            # non e' nell'elenco: al massimo la quota che resta agli altri,
            # anche se l'anno scorso i rigori li batteva lui altrove
            calciati = min(rf[i] + rs[i], calciati)
        rf[i] = calciati * PENALTY_CONVERSION
        rs[i] = calciati * (1.0 - PENALTY_CONVERSION)

    out["p_vote"] = np.clip(p_vote, 0.0, 1.0)
    out["rf"], out["rs"] = rf, rs
    return out
