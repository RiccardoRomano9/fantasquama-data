"""Produce `serieA.json`, il file che l'app iOS legge.

L'app non ricalcola le probabilita': le legge da qui. Ma **non legge i
fantapunti**, perche' quelli dipendono dalle regole della lega dell'utente e
devono essere calcolati sul telefono. Qui escono le probabilita' dei singoli
eventi; l'app le moltiplica per i bonus configurati in Impostazioni.

E' la stessa separazione della spec 5: predizione e punteggio sono due pezzi,
e solo il secondo conosce il regolamento.

    python export_app_data.py --gameweek 38 --season 2025-26

La giornata scelta e' quella "da giocare": la storia usata e' tutta e sola
quella precedente, esattamente come sara' nell'app.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fantasquama import fantaplayer, learned, lineups, odds, pipeline, roster, simulate
from fantasquama.estimate import fit_shrinkage, previous_season
from fantasquama.features import rolling_history
from fantasquama.fixtures import (
    MATCH_FACTOR_MAX,
    MATCH_FACTOR_MIN,
    TEAM_ALIASES,
    attach,
    fit_difficulty,
    load_fixtures,
)
from fantasquama.ingest import CANONICAL_COLUMNS, load_archive
from fantasquama.scoring import EVENTS, Rules, fantavoto


def main() -> None:
    parser = argparse.ArgumentParser(description="Esporta i dati per l'app iOS")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--gameweek", type=int, default=38)
    parser.add_argument("--out", type=Path, default=Path("../ios/FantaSquama/Resources/serieA.json"))
    parser.add_argument(
        "--listone",
        type=Path,
        help="il listone .xlsx di Fantacalcio.it. Con questo l'export descrive la "
             "rosa della stagione da giocare invece dell'ultima giornata archiviata.",
    )
    parser.add_argument(
        "--roster-snapshot",
        type=Path,
        help="un precedente serieA.json da cui riprendere la rosa quando arriva una "
             "nuova giornata di voti ma non serve riscaricare il listone.",
    )
    parser.add_argument("--overrides", type=Path, default=Path("roster-overrides.csv"))
    parser.add_argument(
        "--probabili",
        type=Path,
        help="le probabili formazioni prodotte da fetch_lineups.py. Sostituiscono "
             "la deduzione sulla titolarita' con l'informazione.",
    )
    parser.add_argument(
        "--without-recent-form",
        action="store_true",
        help="non esporta gli ultimi fantavoti reali (per il repository pubblico)",
    )
    parser.add_argument("--piazzati", type=Path, default=Path("set-pieces-2026-27.csv"))
    parser.add_argument("--mantra-history", type=Path, default=Path("data/mantra/quotazioni-2025-26.xlsx"))
    parser.add_argument(
        "--nomi", type=Path, default=Path("nomi-estesi.csv"),
        help="dal listone_id al nome per esteso: il listone scrive «Martinez L.» "
             "e nessuno cerca cosi'",
    )
    parser.add_argument(
        "--fantaplayer", type=Path, default=Path("data/fantaplayer"),
        help="cartella con gli storici stagionali FantaPlayer; rafforza il prior "
             "personale senza duplicare l'archivio giornata per giornata",
    )
    args = parser.parse_args()
    if args.listone and args.roster_snapshot:
        raise SystemExit("usa --listone oppure --roster-snapshot, non entrambi")

    archive = load_archive(args.data)
    if args.mantra_history.exists():
        storico = roster.match(roster.load_listone(args.mantra_history), archive, roster.load_overrides(args.overrides))
        mantra = storico[storico["player_id"] != ""].set_index("player_id")["mantra_role"]
        archive["mantra_role"] = archive["player_id"].map(mantra).fillna("")
    rosa = None
    if args.listone:
        rosa = roster.match(
            roster.load_listone(args.listone), archive, roster.load_overrides(args.overrides)
        )
        archive = _con_la_rosa(archive, rosa, args.season, args.gameweek)
    elif args.roster_snapshot:
        rosa = roster.match(
            roster.load_roster_snapshot(args.roster_snapshot), archive,
            roster.load_overrides(args.overrides),
        )
        archive = _con_la_rosa(archive, rosa, args.season, args.gameweek)
    history = rolling_history(archive)
    fixtures = load_fixtures(args.data / "fixtures")

    target = ((archive["season"] == args.season) & (archive["gameweek"] == args.gameweek)).to_numpy()
    if not target.any():
        raise SystemExit(f"nessuna riga per {args.season} giornata {args.gameweek}")

    # taratura su tutto cio' che precede la giornata da giocare, mai su di essa
    train = (
        (archive["season"] != args.season)
        | (archive["gameweek"] < args.gameweek)
    ).to_numpy()

    previous = previous_season(archive)
    if rosa is not None and args.fantaplayer.exists():
        previous = fantaplayer.enrich_previous(previous, archive, rosa, args.fantaplayer)

    context = attach(archive, fixtures)
    # Qui, a differenza del backtest, non c'e' nessuna stagione da tenere
    # fuori: si prevede la prossima giornata e tutto il resto e' gia'
    # successo, quindi la difficolta' si tara su tutto il passato che c'e'.
    difficulty = fit_difficulty(fixtures, sorted(archive["season"].unique()))
    market = odds.fit(args.data / "fixtures", fixtures)
    formazione = lineups.formazioni(args.probabili, args.piazzati, rosa, archive)
    lambdas = odds.align(market, archive)
    # gli stessi fattori che la pipeline applica: l'app li riceve per poter
    # mostrare quanto di una correzione e' mercato e quanto profilo di squadra
    fattori = pipeline.match_factors(context, difficulty, lambdas, market.league_goals)

    stima = pipeline.estimate(
        archive, history, previous, train,
        context=context, difficulty=difficulty, formazione=formazione,
        market=market,
        # quanto fidarsi del dato personale, stimato su tutto cio' che
        # precede la giornata da giocare: la stessa maschera del calibratore
        shrinkage=fit_shrinkage(archive, train),
    )
    probabilities, votes = stima.probabilities, stima.votes

    # Secondo parere: lo stesso output, imparato dai dati invece che costruito
    # a mano. Sul banco di prova l'insieme dei due batte entrambi, e il loro
    # disaccordo e' una misura onesta di incertezza -- l'app la usa.
    #
    # Le feature sono `stima.blended`, le stesse su cui il modello viene
    # misurato: con la storia grezza, a inizio stagione le medie sono zeri e
    # il modello addestrato qui non sarebbe quello del banco di prova.
    features = learned.build_features(stima.blended, archive, context, stima.lambdas)
    apprese = learned.fit(features, archive, train).predict(features)

    # La forbice del fantavoto, sulle sole righe della giornata da giocare:
    # sull'archivio intero sarebbero tre miliardi di campioni. Il regolamento
    # e' quello di default -- l'app ricalcola la media coi bonus dell'utente,
    # ma le probabilita' di sfondare o floppare restano una buona guida anche
    # con pesi un po' diversi.
    scelte = np.flatnonzero(target)
    forbice = simulate.distribution(
        probabilities.iloc[scelte], archive["role"].iloc[scelte],
        votes.iloc[scelte], Rules(),
    )
    forbice.index = archive.index[scelte]

    forma = _recent_form(archive, args.season, args.gameweek)
    estesi = _nomi_estesi(args.nomi)

    # dal player_id dell'archivio ai dati del listone, per le sole righe che
    # dal listone vengono
    listino = (
        {
            (r.player_id if r.player_id else f"L{r.listone_id}"): r
            for r in rosa.itertuples()
        }
        if rosa is not None
        else {}
    )

    players = []
    for i in np.flatnonzero(target):
        riga, storia, prob = archive.iloc[i], history.iloc[i], probabilities.iloc[i]
        pid = str(riga["player_id"])
        quotato = listino.get(pid)
        listone_id = str(quotato.listone_id) if quotato else ""
        players.append({
            "id": quotato.listone_id if quotato else pid,
            "name": str(riga["player_name"]),
            "fullName": _clean(estesi.get(listone_id) or getattr(quotato, "fullName", None)) if quotato else None,
            "role": str(riga["role"]),
            "mantraRole": str(quotato.mantra_role) if quotato is not None and pd.notna(quotato.mantra_role) else None,
            "team": str(riga["team"]),
            "opponent": _clean(context["opponent"].iloc[i]),
            "home": bool(context["home"].iloc[i]) if pd.notna(context["home"].iloc[i]) else None,
            "winProbability": _round(context["p_win"].iloc[i], 3),
            "drawProbability": _round(context["p_draw"].iloc[i], 3),
            "appearances": int(storia["apps_before"] or 0),
            "gameweeksElapsed": int(storia["gw_elapsed"] or 0),
            "estimatedVote": _round(votes.iloc[i], 2),
            "playProbability": _round(prob["p_vote"], 3),
            "events": {name: _round(prob[name], 4) for name in EVENTS},
            # Due giocatori con gli stessi punti attesi sono decisioni
            # diverse se uno e' regolare e l'altro alterna 4,5 e 9,5.
            "outlook": {
                "high": _round(forbice["p_high"].loc[archive.index[i]], 3),
                "low": _round(forbice["p_low"].loc[archive.index[i]], 3),
                "q10": _round(forbice["q10"].loc[archive.index[i]], 2),
                "q50": _round(forbice["q50"].loc[archive.index[i]], 2),
                "q90": _round(forbice["q90"].loc[archive.index[i]], 2),
            },
            "learnedVote": _round(apprese["voto"].iloc[i], 2),
            "learnedPlayProbability": _round(apprese["p_vote"].iloc[i], 3),
            "learnedEvents": {name: _round(apprese[name].iloc[i], 4) for name in EVENTS},
            "photoURL": _clean(getattr(quotato, "photoURL", None)) if quotato else None,
            "photoProviderID": _clean(getattr(quotato, "photoProviderID", None)) if quotato else None,
            "teamGoalsRate": _round(storia["team_goals_rate"], 2),
            "recentForm": forma.get(pid, []),
            "matchContext": {
                "attack": _round(fattori.attack[i], 5),
                "defense": _round(fattori.defense[i], 5),
                "marketAttack": _round(fattori.market_attack[i], 5),
                "marketDefense": _round(fattori.market_defense[i], 5),
                "hadMarket": bool(fattori.has_market[i]),
            },
            # Il listone e' la sola fonte del prezzo, e per chi in Serie A non
            # ha mai giocato e' anche la sola informazione che esista: l'app
            # deve poter dire "di questo non so niente" invece di dare una
            # media di ruolo con l'aria di una stima.
            "quotazione": _round(quotato.quotazione, 1) if quotato else None,
            "fvm": _round(quotato.fvm, 1) if quotato else None,
            # Cio' che dice la probabile formazione, per esteso. Il rigorista
            # entra gia' nei numeri; i calci da fermo no -- senza gli elenchi
            # delle stagioni passate non c'e' modo di misurare quanto valgano,
            # e in questo progetto non entra niente che non sia misurato.
            "lineupSlot": _clean(formazione["slot"].iloc[i]),
            "startingProbability": _round(formazione["titolarita"].iloc[i], 0),
            "status": _clean(formazione["stato"].iloc[i]) or None,
            "penaltyRank": _int(formazione["rigori"].iloc[i]),
            "setPieceRank": _int(formazione["fermo"].iloc[i]),
            # La confidenza deve riflettere tutto lo storico usato davvero
            # dalla stima, non soltanto le presenze dell'anno in corso o le
            # cinque gare mostrate nell'interfaccia. `previous` comprende sia
            # la Serie A precedente sia l'eventuale profilo FantaPlayer.
            "hasHistory": _has_history(storia["apps_before"], previous["apps_prev"].iloc[i]),
        })

    players.sort(key=lambda p: (p["role"], p["name"]))
    partite = _calendario(args.data / "fixtures", args.season)
    squadre = _squadre(args.probabili, rosa)
    notizie = _notizie(args.data / "news.json")
    odds_updated_at = _odds_updated_at(args.data / "fixtures", args.season)
    _copia_stemmi(args.data / "fixtures" / "crests", args.out.parent / "crests")
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": args.season,
        "gameweek": args.gameweek,
        "note": (
            f"Stime calcolate con la sola storia precedente alla giornata "
            f"{args.gameweek} della stagione {args.season}. Ogni giocatore porta "
            f"due stime indipendenti: una costruita a mano e una imparata dai "
            f"dati. L'app usa la loro media, e il loro disaccordo abbassa la "
            f"confidenza del consiglio. Lo storico stagionale FantaPlayer, se presente, "
            f"rafforza con peso decrescente il prior personale."
        ),
        "players": players,
        "matches": partite,
        "teams": squadre,
        "news": notizie,
        "marketDifficulty": {
            "version": 1,
            "attack": {
                "slope": _round(difficulty.attack[0], 8),
                "intercept": _round(difficulty.attack[1], 8),
                "mean": _round(difficulty.mean_attack, 8),
            },
            "defense": {
                "slope": _round(difficulty.defense[0], 8),
                "intercept": _round(difficulty.defense[1], 8),
                "mean": _round(difficulty.mean_defense, 8),
            },
            "limits": {"min": MATCH_FACTOR_MIN, "max": MATCH_FACTOR_MAX},
        },
    }
    if odds_updated_at:
        payload["oddsUpdatedAt"] = odds_updated_at
        payload["oddsSource"] = "the-odds-api"
    if args.without_recent_form:
        for player in payload["players"]:
            player.pop("recentForm", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"{args.out}: {len(players)} giocatori, {len(partite)} partite, "
          f"{args.out.stat().st_size:,} byte")


def _calendario(root: Path, season: str) -> list[dict]:
    """Tutte le partite della stagione, non solo quelle della giornata.

    Sono tutte e 380 perche' la scheda «Partite» decide da sola quale
    giornata mostrare, in base al giorno in cui la si apre: mandarle tutte
    costa una manciata di kilobyte e toglie all'app la dipendenza da un
    export fatto al momento giusto.
    """
    anno = int(season[:4])
    percorso = root / f"matches_{anno}.json"
    if not percorso.exists():
        return []

    def squadra(lato: dict) -> str:
        nome = str(lato.get("shortName") or lato.get("name") or "")
        return TEAM_ALIASES.get(nome, nome)

    partite = []
    for match in json.loads(percorso.read_text())["matches"]:
        punteggio = match["score"]["fullTime"]
        partite.append({
            "matchday": int(match["matchday"]),
            "date": match["utcDate"],
            "status": match["status"],
            "home": squadra(match["homeTeam"]),
            "away": squadra(match["awayTeam"]),
            "homeCode": match["homeTeam"].get("tla"),
            "awayCode": match["awayTeam"].get("tla"),
            "homeGoals": punteggio.get("home"),
            "awayGoals": punteggio.get("away"),
        })
    partite.sort(key=lambda p: (p["matchday"], p["date"]))
    return partite


def _notizie(path: Path) -> list[dict]:
    """Le notizie prodotte da `fetch_news.py`, se ci sono.

    Facoltative di proposito: senza, la scheda «Home» lo dice e il resto
    dell'app non se ne accorge. Un giornale che non risponde non deve poter
    impedire di sapere chi schierare.
    """
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _odds_updated_at(root: Path, season: str) -> str | None:
    path = root / f"odds_{season[:4]}.csv"
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None
    if "LastUpdate" not in raw.columns:
        return None
    values = [
        str(value).strip()
        for value in raw["LastUpdate"].dropna()
        if str(value).strip()
    ]
    return max(values) if values else None


def _copia_stemmi(sorgente: Path, destinazione: Path) -> None:
    """Gli stemmi nel bundle dell'app, che di rete non ne fa.

    Sono immagini di terzi: come `serieA.json` non stanno in git, e chi
    clona il progetto se li riscarica con `fetch_fixtures.py`.
    """
    if not sorgente.exists():
        return
    destinazione.mkdir(parents=True, exist_ok=True)
    for stemma in sorted(sorgente.glob("*.png")):
        destinazione.joinpath(stemma.name).write_bytes(stemma.read_bytes())


def _squadre(path: Path | None, rosa: pd.DataFrame | None) -> list[dict]:
    """Modulo e ballottaggi di ogni squadra, per le probabili formazioni."""
    if path is None:
        return []
    _, squadre = lineups.load_probabili(path)
    if rosa is not None:
        squadre = lineups.enrich_ballottaggi(squadre, rosa)
    return [
        {
            # stesso alias del calendario: «Como» e «Como 1907» devono essere
            # la stessa squadra, o a schermo il modulo sparisce
            "name": TEAM_ALIASES.get(str(r.squadra), str(r.squadra)),
            "formation": str(r.modulo),
            "isOfficial": bool(r.ufficiale),
            "ballottaggi": r.ballottaggi,
        }
        for r in squadre.itertuples()
    ]


def _con_la_rosa(
    archive: pd.DataFrame, rosa: pd.DataFrame, season: str, gameweek: int
) -> pd.DataFrame:
    """Aggiunge all'archivio una riga vuota per ogni giocatore del listone.

    Per la giornata da giocare le righe sono segnaposto, con `played` falso e
    nessun evento. Servono solo a far esistere ogni giocatore nella pipeline,
    che da li' in poi non deve sapere niente di listoni. Se la stagione e'
    gia' iniziata, le giornate precedenti restano nell'archivio e aggiornano
    la storia: e' cosi' che il confronto passa davvero alla giornata seguente.

    L'identita' che porta la storia e' il `player_id` dell'archivio. Chi non
    si e' agganciato ne riceve uno tutto suo, ricavato dal codice del listone:
    esiste nella rosa, ma per il modello e' un esordiente, perche' lo e'.
    """
    target = (archive["season"] == season) & (archive["gameweek"] == gameweek)
    if target.any():
        raise SystemExit(
            f"la giornata {gameweek} di {season} e' gia' in archivio: scegli la "
            "prossima giornata, non una gia' giocata."
        )
    righe = pd.DataFrame({
        "season": season,
        "gameweek": gameweek,
        "player_id": rosa["player_id"].where(
            rosa["player_id"] != "", "L" + rosa["listone_id"]
        ),
        "player_name": rosa["player_name"],
        "role": rosa["role"],
        "mantra_role": rosa["mantra_role"],
        "team": rosa["team"],
        "played": False,
        "voto": np.nan,   # non ha ancora giocato: non e' un dato mancante, e' il futuro
        **{event: 0 for event in EVENTS},
    })
    unito = pd.concat([archive, righe], ignore_index=True)
    return unito.astype({c: t for c, t in CANONICAL_COLUMNS.items() if c in unito})


def _nomi_estesi(path: Path) -> dict[str, str]:
    """Dal `listone_id` al nome per esteso. Vuoto se il file non c'e'.

    Non e' un dato indispensabile: senza, la ricerca resta quella di prima e
    l'app funziona uguale. Serve perche' nessuno cerca «Martinez L.»: si
    cerca «Lautaro».
    """
    if not path.exists():
        return {}
    tabella = pd.read_csv(path, dtype="string")
    return {
        str(r.listone_id): str(r.esteso)
        for r in tabella.itertuples()
        if pd.notna(r.esteso)
    }


def _recent_form(archive: pd.DataFrame, season: str, gameweek: int) -> dict[str, list[float | None]]:
    """Il fantavoto di ogni giornata di questa stagione, dalla G1 all'ultima
    gia' giocata: un elemento per giornata, `None` per chi quella giornata
    non ha preso voto.

    La posizione nell'array E' la giornata (indice 0 = G1): niente da
    dedurre a valle contando quante ne sono passate. E' il motivo per cui non
    si scarta chi non ha preso voto -- lo si lascia `None` al posto suo --
    invece di saltarlo: saltarlo sfaserebbe ogni giornata successiva di una
    posizione.
    """
    rules = Rules()
    corrente = archive[(archive["season"] == season) & (archive["gameweek"] < gameweek)]
    out: dict[str, list[float | None]] = {}
    for pid, gruppo in corrente.groupby("player_id", observed=True):
        per_giornata = {
            int(riga.gameweek): (
                round(fantavoto(float(riga.voto), {e: getattr(riga, e) for e in EVENTS}, rules), 1)
                if riga.played else None
            )
            for riga in gruppo.itertuples()
        }
        out[str(pid)] = [per_giornata.get(g) for g in range(1, gameweek)]
    return out


def _has_history(current_apps: object, previous_apps: object) -> bool:
    """Vero se la stima dispone di almeno una presenza personale reale."""
    values = pd.to_numeric(pd.Series([current_apps, previous_apps]), errors="coerce")
    return bool((values.fillna(0.0) > 0).any())


def _int(value):
    return None if pd.isna(value) else int(value)


def _round(value, digits: int = 2):
    return None if pd.isna(value) else round(float(value), digits)


def _clean(value):
    return None if pd.isna(value) else str(value)


if __name__ == "__main__":
    main()
