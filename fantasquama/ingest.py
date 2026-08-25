"""Archivio storico dei voti, portato allo schema canonico.

Una riga per giocatore per giornata. Il fantavoto dell'archivio non viene
importato: lo ricalcoliamo dagli eventi con scoring.fantavoto.

**Se SOURCE_COLUMNS e' sbagliata, il backtest gira lo stesso e produce un
numero credibile.** Ogni errore di mappatura fallisce in silenzio: `Voto` su
una colonna di testo rende tutti SV, una colonna evento su una colonna di
testo la azzera, `Ruolo` sbagliata svuota il frame. Non esiste una verifica
generale possibile -- l'archivio non espone un fantavoto gia' calcolato con
cui riconciliare -- quindi `normalize` controlla le due cose che nessun
archivio reale puo' violare:

- almeno una riga con un ruolo riconosciuto,
- almeno un giocatore che ha preso voto.

Una giornata dove nessuno ha giocato o nessuno ha un ruolo valido significa
mappatura sbagliata, non dati insoliti. Restano fuori dalla rete gli errori
piu' sottili (una colonna evento scambiata con un'altra): l'unico rimedio
resta guardare le intestazioni del file, come dice il commento su
SOURCE_COLUMNS.
"""

import re
from pathlib import Path

import pandas as pd

from fantasquama.scoring import EVENTS

ROLES = ("P", "D", "C", "A")

# `cs` -- porta inviolata -- non e' una colonna dell'archivio: e' una domanda
# che si fa alle colonne che ci sono. Vale 1 quando un portiere ha preso voto
# e non ha subito gol, 0 in ogni altro caso, portiere o no.
DERIVED_EVENTS: tuple[str, ...] = ("cs",)
SOURCE_EVENTS: tuple[str, ...] = tuple(e for e in EVENTS if e not in DERIVED_EVENTS)

# Solo le cartelle con questo nome sono stagioni. Accettare qualunque cartella
# significherebbe provare a leggere come voti tutto quello che finisce sotto
# `data/` -- il calendario, una cartella temporanea, un residuo di sistema.
SEASON_DIR = re.compile(r"\d{4}-\d{2}")

CANONICAL_COLUMNS: dict[str, str] = {
    "season": "string",
    "gameweek": "int16",
    "player_id": "string",
    "player_name": "string",
    "role": "string",
    "team": "string",
    "played": "bool",
    "voto": "float32",
    **{event: "int8" for event in EVENTS},
}

# Mappatura dalle intestazioni dell'archivio ai nomi canonici.
#
# Verificata sull'export .xls per giornata di fanta.soccer / Fantacalcio.it
# (stagioni 2024-25 e 2025-26). Intestazioni sulla prima riga, nessuna riga
# di testata da saltare.
#
# Se le intestazioni differiscono, correggere QUI: e' l'unico punto del
# progetto che conosce il formato della sorgente.
SOURCE_COLUMNS: dict[str, str] = {
    "Codice": "player_id",
    "Ruolo": "role",
    "Squadra": "team",
    "Voto FS": "voto",
    "Gol_segnati_fs": "gf",
    "Gol_subiti_fs": "gs",
    "Rigori_parati": "rp",
    "Rigori_sbagliati": "rs",
    "Rigori_segnati": "rf",
    "Autorete_fs": "au",
    "Ammonizione": "amm",
    "Espulsione": "esp",
    "Assist_fs": "ass",
}

# Il nome del giocatore sta su due colonne; `Nome` e' spesso vuoto.
NAME_COLUMNS: tuple[str, str] = ("Cognome", "Nome")

# La giornata e' dichiarata anche dentro il file: la usiamo per verificare
# quella dedotta dal percorso, non per sostituirla.
GAMEWEEK_COLUMN = "Giornata"


def _numeric(column: pd.Series) -> pd.Series:
    """`pd.to_numeric` che capisce la virgola decimale italiana.

    Un archivio italiano scrive "6,5". `pd.to_numeric(errors="coerce")` secco
    lo trasforma in NaN, che qui significa "non ha preso voto": un SV
    silenzioso su ogni riga della colonna. Le colonne gia' numeriche passano
    intatte, quindi la sostituzione non tocca mai un valore valido.
    """
    if not pd.api.types.is_numeric_dtype(column):
        column = column.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(column, errors="coerce")


def normalize(raw: pd.DataFrame, season: str, gameweek: int) -> pd.DataFrame:
    """Porta una giornata grezza allo schema canonico.

    Solleva se la giornata esce senza nessuna riga con ruolo valido o senza
    nessun giocatore che ha preso voto: vedi il docstring del modulo.
    """
    missing = (set(SOURCE_COLUMNS) | set(NAME_COLUMNS)) - set(raw.columns)
    if missing:
        raise ValueError(f"colonne mancanti nell'archivio: {sorted(missing)}")

    # La giornata arriva dal percorso; il file la dichiara anche al proprio
    # interno. Se le due non coincidono il file e' stato messo nella cartella
    # sbagliata, e ogni media a valle sarebbe calcolata su giornate mescolate.
    if GAMEWEEK_COLUMN in raw.columns:
        dichiarate = pd.to_numeric(raw[GAMEWEEK_COLUMN], errors="coerce").dropna()
        dichiarate = {int(x) for x in dichiarate.unique()}
        if dichiarate and dichiarate != {gameweek}:
            raise ValueError(
                f"{season}: il file dichiara giornata {sorted(dichiarate)} "
                f"ma viene caricato come giornata {gameweek}."
            )

    df = raw[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS).copy()
    df["season"] = season
    df["gameweek"] = gameweek

    cognome, nome = (raw[c].astype("string").str.strip() for c in NAME_COLUMNS)
    df["player_name"] = (cognome.fillna("") + " " + nome.fillna("")).str.strip()
    df["team"] = df["team"].astype("string").str.strip()
    df["role"] = df["role"].astype("string").str.strip().str.upper()
    df = df[df["role"].isin(ROLES)]
    if df.empty:
        raise ValueError(
            f"{season} giornata {gameweek}: nessuna riga con un ruolo fra {list(ROLES)}. "
            f"Controllare la voce 'Ruolo' di SOURCE_COLUMNS."
        )

    # L'id passa per Int64 prima di diventare stringa. `Cod.` viene letto
    # int64 se la colonna e' pulita e float64 se una sola cella e' vuota
    # (comunissimo: righe di piede pagina): `astype("string")` diretto darebbe
    # "2101" in una giornata e "2101.0" in un'altra, cioe' due giocatori
    # distinti per ogni raggruppamento a valle, senza errore ne' avviso.
    # errors="raise" perche' una colonna non numerica qui significa
    # SOURCE_COLUMNS sbagliata, non un dato insolito.
    # La conversione viene dopo il filtro sui ruoli, cosi' le righe di piede
    # pagina (senza ruolo valido) sono gia' fuori e un id mancante fra le
    # righe rimaste e' davvero un difetto dell'archivio.
    df["player_id"] = pd.to_numeric(df["player_id"], errors="raise").astype("Int64").astype("string")
    if df["player_id"].isna().any():
        raise ValueError(
            f"{season} giornata {gameweek}: righe con ruolo valido ma senza Cod."
        )

    # Nell'archivio chi non scende in campo ha voto assente o zero.
    df["voto"] = _numeric(df["voto"])
    df["played"] = df["voto"].notna() & (df["voto"] > 0)
    df.loc[~df["played"], "voto"] = pd.NA
    if not df["played"].any():
        raise ValueError(
            f"{season} giornata {gameweek}: nessun giocatore ha preso voto. "
            f"Controllare la voce 'Voto' di SOURCE_COLUMNS."
        )

    for event in SOURCE_EVENTS:
        df[event] = _numeric(df[event]).fillna(0)

    # `Gol_segnati_fs` include gia' i rigori: verificato sull'archivio reale
    # (Zaccagni 24-25 giornata 1: Voto 7, Gol 1, Rigori 1 -- quel gol E' il
    # rigore). Il modello li tiene separati e li premia entrambi, quindi
    # mappare la colonna dritta su `gf` darebbe +6 per una rete sola e
    # spingerebbe ogni consiglio verso i rigoristi.
    if (df["rf"] > df["gf"]).any():
        raise ValueError(
            f"{season} giornata {gameweek}: piu' rigori segnati che gol totali. "
            f"'Gol_segnati_fs' non include piu' i rigori, oppure SOURCE_COLUMNS "
            f"e' sbagliata: la sottrazione qui sotto darebbe gol negativi."
        )
    df["gf"] = df["gf"] - df["rf"]

    # La porta inviolata e' un fatto del portiere, non della squadra: se il
    # portiere e' entrato a partita in corso e ha preso gol, quella partita
    # non e' inviolata per lui anche se la squadra ne ha subito uno solo.
    df["cs"] = ((df["role"] == "P") & df["played"] & (df["gs"] == 0)).astype("int8")

    df = df[list(CANONICAL_COLUMNS)].astype(CANONICAL_COLUMNS)
    return df.reset_index(drop=True)


def load_gameweek(path: Path, season: str, gameweek: int) -> pd.DataFrame:
    """Legge un file di giornata e lo normalizza."""
    raw = pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)
    return normalize(raw, season=season, gameweek=gameweek)


def _gameweek(path: Path) -> int:
    """Numero di giornata dedotto dal nome del file.

    Regge sia `1.xlsx` sia `Voti_1a_SerieA.xls`. Solleva se i numeri sono
    zero o piu' di uno, invece di sceglierne uno: indovinare qui vorrebbe
    dire attribuire un'intera giornata alla giornata sbagliata, in silenzio.
    """
    numbers = re.findall(r"\d+", path.stem)
    if len(numbers) != 1:
        raise ValueError(
            f"{path.name}: impossibile dedurre la giornata dal nome del file "
            f"(numeri trovati: {numbers}). Ne serve esattamente uno."
        )
    return int(numbers[0])


def load_archive(root: Path) -> pd.DataFrame:
    """Carica tutte le giornate sotto `root`.

    Struttura attesa:  root/<stagione>/<qualsiasi nome con un numero>.<xlsx|xls|csv>
    dove <stagione> e' del tipo "2024-25". Le altre cartelle sotto `root` -- il
    calendario, per esempio -- vengono ignorate.
    dove <stagione> e' del tipo "2024-25" e <giornata> e' un numero.

    Una riga per giocatore per giornata: la chiave (stagione, giocatore,
    giornata) dev'essere unica. Un duplicato -- lo stesso giocatore due volte
    nello stesso file, o due file per la stessa giornata -- gonfia
    `gw_elapsed` e conta due volte la presenza, quindi falsa ogni media senza
    dare nessun segnale.
    """
    stagioni = sorted(p for p in root.iterdir() if p.is_dir() and SEASON_DIR.fullmatch(p.name))
    if not stagioni:
        altre = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise ValueError(
            f"nessuna cartella stagione sotto {root}: servono nomi come '2024-25'. "
            f"Cartelle trovate: {altre or 'nessuna'}"
        )

    frames = []
    for season_dir in stagioni:
        for path in sorted(season_dir.iterdir()):
            if path.suffix not in {".xlsx", ".xls", ".csv"}:
                continue
            frames.append(load_gameweek(path, season_dir.name, _gameweek(path)))
    if not frames:
        raise ValueError(f"nessuna giornata trovata sotto {root}")

    archive = pd.concat(frames, ignore_index=True)
    key = ["season", "player_id", "gameweek"]
    # mascheramento posizionale, non `.loc` per etichetta: la convenzione del
    # progetto, cosi' l'indice non puo' mai entrare nel risultato
    duplicated = archive.duplicated(subset=key, keep=False).to_numpy()
    if duplicated.any():
        esempi = archive[duplicated][key].drop_duplicates().head(5)
        raise ValueError(
            f"{int(duplicated.sum())} righe duplicate su (stagione, giocatore, "
            f"giornata); prime chiavi coinvolte:\n{esempi.to_string(index=False)}"
        )
    return archive
