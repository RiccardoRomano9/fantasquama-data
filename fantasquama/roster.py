"""Il listone di Fantacalcio.it, agganciato all'archivio dei voti.

Il listone dice chi c'e' in Serie A **quest'anno**, con ruolo e quotazione.
L'archivio dice come ha giocato **l'anno scorso**. Sono due fonti diverse con
due spazi di identificatori diversi -- il listone numera i giocatori a quattro
cifre, l'archivio a sette -- quindi l'unico ponte fra le due e' il nome.

I nomi non combaciano mai del tutto. Il listone scrive `Gila`, l'archivio
`Mario Gila`; il listone `Hojlund`, l'archivio `Højlund R.`; e quando due
giocatori hanno lo stesso cognome il listone aggiunge un'iniziale
(`Martinez Jo.`) che l'archivio scrive in un altro modo. Qui si confrontano
quindi gli *insiemi di parole* del nome, non le stringhe: `{gila}` sta dentro
`{mario, gila}`, e tanto basta. Chi resta ambiguo si decide con squadra,
ruolo e presenze; chi resta ambiguo anche cosi' si scrive a mano in
`roster-overrides.csv`, che sta in git proprio perche' e' una decisione, non
un dato.

Chi non si aggancia NON e' un errore: sono i neopromossi e gli acquisti
dall'estero, che in Serie A non hanno mai giocato. Per loro non esiste storia
da cui stimare, e l'app deve saperlo.

Nessuna rete: il file lo si scarica a mano da Fantacalcio.it.
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Il file .xlsx ha un titolo sulla prima riga e le intestazioni sulla seconda.
HEADER_ROW: int = 1

# Dalle intestazioni del listone ai nomi canonici. Il resto delle colonne
# Il Mantra non sostituisce `R`: aggiunge la posizione reale al modello.
LISTONE_COLUMNS: dict[str, str] = {
    "Id": "listone_id",
    "R": "role",
    "RM": "mantra_role",
    "Nome": "player_name",
    "Squadra": "team",
    "Qt.A": "quotazione",
    "FVM": "fvm",
}

# Il listone abbrevia dove l'archivio no. E' l'unica differenza fra le due
# liste di squadre: le altre diciannove combaciano parola per parola.
TEAM_ALIASES: dict[str, str] = {"Como": "Como 1907"}

# Sotto questa quota di agganci la mappatura si e' rotta -- intestazioni
# cambiate, archivio vuoto, listone di un'altra lega -- e proseguire
# significherebbe costruire una rosa di sconosciuti senza che nulla lo dica.
# Il valore misurato sul listone 2026-27 e' 0,84.
MIN_MATCH_RATE: float = 0.70

# Lettere che NFKD non scompone: la Ø danese non e' una O con un segno sopra,
# e' un carattere a se'. Senza questa tabella `Højlund` diventa `hjlund` e non
# aggancia niente.
_TRANSLITERATION = {
    "ø": "o", "đ": "d", "ð": "d", "ł": "l", "ı": "i",
    "æ": "ae", "œ": "oe", "ß": "ss", "þ": "th",
}

_INITIAL = re.compile(r"\s([A-Za-z]{1,4})\.\s*$")


def name_tokens(name: object) -> tuple[str, ...]:
    """Le parole del nome, normalizzate e ordinate.

    Ordinate perche' le due fonti mettono nome e cognome in ordine diverso:
    l'archivio scrive `Mario Gila`, il listone `Gila`. Le iniziali puntate
    spariscono qui e si recuperano con `name_initial`.
    """
    # via l'iniziale puntata PRIMA di spezzare: `Jo.` diventerebbe altrimenti
    # la parola `jo`, che nessun nome per esteso contiene, e il confronto fra
    # insiemi fallirebbe proprio sui nomi che l'iniziale serve a distinguere
    text = _INITIAL.sub("", str(name))
    for source, target in _TRANSLITERATION.items():
        text = text.replace(source, target).replace(source.upper(), target)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[^a-z ]", " ", text.replace("'", " ").replace("-", " "))
    return tuple(sorted(t for t in text.split() if len(t) > 1))


def name_initial(name: object) -> str:
    """`'Martinez Jo.'` -> `'jo'`. Stringa vuota se il nome non ne porta."""
    found = _INITIAL.search(str(name))
    return found.group(1).lower() if found else ""


def load_listone(path: Path) -> pd.DataFrame:
    """Il listone ufficiale, ridotto alle colonne che servono."""
    raw = pd.read_excel(path, skiprows=HEADER_ROW)
    missing = sorted(set(LISTONE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(
            f"il listone non ha le colonne {missing}: le intestazioni sono "
            f"{sorted(map(str, raw.columns))}. Correggere LISTONE_COLUMNS."
        )
    out = raw[list(LISTONE_COLUMNS)].rename(columns=LISTONE_COLUMNS)
    out["listone_id"] = out["listone_id"].astype("int64").astype("string")
    out["role"] = out["role"].astype("string")
    out["mantra_role"] = out["mantra_role"].astype("string")
    out["player_name"] = out["player_name"].astype("string")
    out["team"] = out["team"].astype("string").replace(TEAM_ALIASES)
    out["quotazione"] = pd.to_numeric(out["quotazione"], errors="raise").astype("float32")
    out["fvm"] = pd.to_numeric(out["fvm"], errors="raise").astype("float32")
    return out.reset_index(drop=True)


def load_roster_snapshot(path: Path) -> pd.DataFrame:
    """Legge la rosa dall'ultimo `serieA.json` esportato.

    Il listone resta la sorgente primaria. Questo ingresso serve quando e'
    arrivata una nuova giornata di voti ma il suo file non e' piu' a portata:
    l'export precedente contiene gia' gli stessi id, ruoli, squadre e prezzi
    necessari per costruire la giornata successiva. Non recupera dati da
    rete e controlla esplicitamente il contratto invece di indovinare.
    """
    data = json.loads(Path(path).read_text())
    players = pd.DataFrame(data.get("players") or [])
    source = {"id", "name", "role", "team", "quotazione", "fvm"}
    missing = source - set(players.columns)
    if missing:
        raise ValueError(f"{path} non contiene i campi rosa {sorted(missing)}")

    if "mantraRole" not in players: players["mantraRole"] = pd.NA
    for column in ("fullName", "photoURL", "photoProviderID"):
        if column not in players:
            players[column] = pd.NA
    out = players.rename(columns={"id": "listone_id", "name": "player_name", "mantraRole": "mantra_role"})[
        [
            "listone_id", "role", "mantra_role", "player_name", "team",
            "quotazione", "fvm", "fullName", "photoURL", "photoProviderID",
        ]
    ].copy()
    out["listone_id"] = out["listone_id"].astype("string")
    out["role"] = out["role"].astype("string")
    out["mantra_role"] = out["mantra_role"].astype("string")
    out["player_name"] = out["player_name"].astype("string")
    out["team"] = out["team"].astype("string").replace(TEAM_ALIASES)
    for column in ("fullName", "photoURL", "photoProviderID"):
        out[column] = out[column].astype("string")
    for column in ("quotazione", "fvm"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("float32")
    if out["listone_id"].isna().any() or out["listone_id"].duplicated().any():
        raise ValueError(f"{path} porta id del listone mancanti o duplicati")
    return out.reset_index(drop=True)


def load_overrides(path: Path) -> dict[str, str]:
    """Gli agganci decisi a mano: `listone_id` -> `player_id`.

    Un `player_id` vuoto significa "questo giocatore non e' nell'archivio,
    smetti di cercarlo": serve per i casi in cui l'omonimia porterebbe
    l'automatismo ad agganciare la persona sbagliata.
    """
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype="string", comment="#")
    missing = {"listone_id", "player_id"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} deve avere le colonne listone_id e player_id")
    return {
        str(row.listone_id): ("" if pd.isna(row.player_id) else str(row.player_id))
        for row in table.itertuples()
    }


def _archive_players(archive: pd.DataFrame) -> pd.DataFrame:
    """Un giocatore per riga, con la squadra e il ruolo piu' recenti."""
    ordered = archive.sort_values(["season", "gameweek"])
    apps = (
        ordered.assign(_p=ordered["played"].astype(int))
        .groupby("player_id", observed=True)["_p"]
        .sum()
    )
    last = ordered.drop_duplicates("player_id", keep="last")
    out = pd.DataFrame({
        "player_id": last["player_id"].astype(str).to_numpy(),
        "player_name": last["player_name"].astype(str).to_numpy(),
        "team": last["team"].astype(str).to_numpy(),
        "role": last["role"].astype(str).to_numpy(),
        "season": last["season"].astype(str).to_numpy(),
    })
    out["tokens"] = [name_tokens(n) for n in out["player_name"]]
    out["initial"] = [name_initial(n) for n in out["player_name"]]
    out["appearances"] = apps.reindex(last["player_id"]).to_numpy()
    return out


def match(
    listone: pd.DataFrame, archive: pd.DataFrame, overrides: dict[str, str] | None = None
) -> pd.DataFrame:
    """Aggiunge al listone `player_id` e `match_kind`.

    `player_id` e' vuoto per chi in archivio non c'e'. `match_kind` dice come
    si e' arrivati all'aggancio, dal piu' sicuro al meno: e' l'unica cosa che
    permette di controllare a mano i casi dubbi invece di fidarsi in blocco.
    """
    overrides = overrides or {}
    players = _archive_players(archive)

    by_word: dict[str, list[int]] = {}
    by_squash: dict[str, list[int]] = {}
    for i, tokens in enumerate(players["tokens"]):
        for word in tokens:
            by_word.setdefault(word, []).append(i)
        # senza spazi: aggancia `Delprato` a `Del Prato`, che come insiemi di
        # parole sono due cose diverse
        by_squash.setdefault("".join(tokens), []).append(i)

    found: list[str] = []
    kinds: list[str] = []
    for row in listone.itertuples():
        forced = overrides.get(str(row.listone_id))
        if forced is not None:
            found.append(forced)
            kinds.append("manual" if forced else "absent")
            continue

        tokens, initial = name_tokens(row.player_name), name_initial(row.player_name)
        candidates = {i for word in tokens for i in by_word.get(word, [])}
        # uno dei due nomi dev'essere contenuto nell'altro: `{gila}` sta in
        # `{mario, gila}`, ma `{moreno}` e `{alberto, moreno}` restano due
        # candidati distinti, ed e' giusto cosi'
        candidates = {
            i for i in candidates
            if set(tokens) <= set(players["tokens"][i]) or set(players["tokens"][i]) <= set(tokens)
        }
        candidates |= set(by_squash.get("".join(tokens), []))
        subset = players.iloc[sorted(candidates)]
        # l'iniziale esclude, non spareggia: `Adams A.` e `Adams C.` sono due
        # persone anche quando in archivio ce n'e' una sola con quel cognome,
        # e trattarla come spareggio la farebbe vincere a entrambi
        subset = subset[[
            _compatible(row_a, initial, tokens) for row_a in subset.itertuples()
        ]]

        player_id, kind = _disambiguate(subset, str(row.team), str(row.role))
        found.append(player_id)
        kinds.append(kind)

    # Un giocatore dell'archivio non puo' stare dietro a due nomi del listone:
    # se succede, uno dei due agganci e' sbagliato e non si sa quale. Tenerli
    # entrambi darebbe a due giocatori diversi la stessa storia -- l'errore
    # peggiore possibile qui, perche' a valle sembra un dato buono.
    seen: dict[str, list[int]] = {}
    for i, player_id in enumerate(found):
        if player_id:
            seen.setdefault(player_id, []).append(i)
    for shared in seen.values():
        if len(shared) > 1 and "manual" not in [kinds[i] for i in shared]:
            for i in shared:
                found[i], kinds[i] = "", "ambiguous"

    out = listone.copy()
    out["player_id"] = pd.array(found, dtype="string")
    out["match_kind"] = pd.array(kinds, dtype="string")

    rate = float((out["player_id"] != "").mean()) if len(out) else 0.0
    if rate < MIN_MATCH_RATE:
        raise ValueError(
            f"solo il {rate:.0%} del listone si aggancia all'archivio (minimo "
            f"{MIN_MATCH_RATE:.0%}): le intestazioni o i nomi non sono quelli attesi"
        )
    return out


def _compatible(candidate, initial: str, tokens: tuple[str, ...]) -> bool:
    """L'iniziale del listone puo' appartenere a questo giocatore dell'archivio?

    Il confronto avviene contro l'iniziale dell'archivio (`Jo.` contro `J.`)
    oppure contro le parole che l'archivio ha in piu' rispetto al listone,
    che sono il nome per esteso (`Jo.` contro `Josep Martinez`). Se l'archivio
    il nome non lo dice affatto, non si puo' escludere niente: passa.
    """
    if not initial:
        return True
    if candidate.initial:
        return candidate.initial.startswith(initial[:1])
    extra = [w for w in candidate.tokens if w not in tokens]
    return any(w.startswith(initial) for w in extra) if extra else True


def _disambiguate(subset: pd.DataFrame, team: str, role: str) -> tuple[str, str]:
    """Sceglie fra i candidati, dal criterio piu' forte al piu' debole."""
    if subset.empty:
        return "", "absent"

    steps: list[tuple[str, pd.DataFrame]] = [
        ("unique", subset),
        ("team+role", subset[(subset["team"] == team) & (subset["role"] == role)]),
        ("team", subset[subset["team"] == team]),
        ("role", subset[subset["role"] == role]),
    ]
    for kind, selection in steps:
        if len(selection) == 1:
            return str(selection["player_id"].iloc[0]), kind

    # Restano solo doppioni della stessa fonte: lo stesso giocatore con due
    # codici, uno dei quali non ha mai preso voto. Vince chi ha giocato.
    ranked = subset.sort_values("appearances", ascending=False)
    if len(ranked) > 1 and ranked["appearances"].iloc[0] > ranked["appearances"].iloc[1]:
        return str(ranked["player_id"].iloc[0]), "appearances"
    return "", "ambiguous"
