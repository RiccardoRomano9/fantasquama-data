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

from fantasquama import calibrate, fantaplayer, learned, lineups, roster
from fantasquama.estimate import (
    _previous_label,
    apply_match_context,
    blended_history,
    event_probabilities,
    previous_season,
)
from fantasquama.features import rolling_history
from fantasquama.fixtures import TEAM_ALIASES, attach, fit_difficulty, load_fixtures, team_factors
from fantasquama.ingest import CANONICAL_COLUMNS, load_archive
from fantasquama.scoring import EVENTS, Rules, fantavoto

FORM_WINDOW = 5


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
    blended = blended_history(history, previous)
    models = calibrate.fit(
        blended.iloc[train], archive["role"].iloc[train], archive["voto"].iloc[train]
    )
    votes = calibrate.predict(models, blended, archive["role"])
    probabilities = event_probabilities(history, archive["role"], train, previous)

    # Le formazioni entrano PRIMA della difficolta' della partita: i rigori
    # che assegnano sono quelli di una squadra media, e una squadra favorita
    # ne ottiene di piu'. Dopo, la correzione non li toccherebbe.
    formazione = _formazioni(args.probabili, args.piazzati, rosa, archive)
    probabilities = lineups.apply(
        probabilities, archive["role"], archive["team"],
        formazione["slot"], formazione["titolarita"], formazione["rigori"],
    )

    context = attach(archive, fixtures)
    advantage = (context["p_win"] - context["p_lose"]).to_numpy(np.float64)
    market_attack, market_defense = fit_difficulty(fixtures, sorted(archive["season"].unique())).factors(advantage)
    squad_attack, squad_defense = team_factors(context)
    has_market = np.isfinite(advantage)
    # Le quote restano il segnale principale. Senza quote, il profilo squadra
    # diventa il contesto della partita; con quote lo rifinisce appena.
    attack = market_attack * np.where(has_market, squad_attack ** 0.20, squad_attack)
    defense = market_defense * np.where(has_market, squad_defense ** 0.20, squad_defense)
    probabilities = apply_match_context(probabilities, attack, defense)

    # Secondo parere: lo stesso output, imparato dai dati invece che costruito
    # a mano. Sul banco di prova l'insieme dei due batte entrambi, e il loro
    # disaccordo e' una misura onesta di incertezza -- l'app la usa.
    features = learned.build_features(history, archive, context)
    apprese = learned.fit(features, archive, train).predict(features)

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
        players.append({
            "id": quotato.listone_id if quotato else pid,
            "name": str(riga["player_name"]),
            "fullName": estesi.get(str(quotato.listone_id)) if quotato else None,
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
            "learnedVote": _round(apprese["voto"].iloc[i], 2),
            "learnedPlayProbability": _round(apprese["p_vote"].iloc[i], 3),
            "learnedEvents": {name: _round(apprese[name].iloc[i], 4) for name in EVENTS},
            "teamGoalsRate": _round(storia["team_goals_rate"], 2),
            "recentForm": forma.get(pid, []),
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
            "hasHistory": bool(int(storia["apps_before"] or 0)) or (
                quotato is not None and quotato.match_kind not in ("absent", "ambiguous")
                and pid in forma
            ),
        })

    players.sort(key=lambda p: (p["role"], p["name"]))
    partite = _calendario(args.data / "fixtures", args.season)
    squadre = _squadre(args.probabili, rosa)
    notizie = _notizie(args.data / "news.json")
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
    }
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


def _formazioni(
    probabili: Path | None, piazzati: Path | None,
    rosa: pd.DataFrame | None, archive: pd.DataFrame
) -> pd.DataFrame:
    """Le colonne della probabile formazione, allineate all'archivio.

    Senza file, colonne vuote: `lineups.apply` non tocca niente e il modello
    resta quello di prima. E' la scelta giusta -- una formazione vecchia di
    una settimana dice meno delle presenze vere.

    Le due fonti hanno cadenze diverse e stanno in due file: titolari,
    panchina e indisponibili cambiano ogni settimana e li scarica lo scraper;
    rigoristi e battitori da fermo cambiano una volta a stagione e stanno a
    mano.
    """
    vuoto = pd.DataFrame({
        "slot": pd.Series([None] * len(archive), index=archive.index, dtype=object),
        "titolarita": pd.Series(np.nan, index=archive.index, dtype=float),
        "stato": pd.Series([None] * len(archive), index=archive.index, dtype=object),
        "rigori": pd.Series(np.nan, index=archive.index, dtype=float),
        "fermo": pd.Series(np.nan, index=archive.index, dtype=float),
    })
    if probabili is None:
        return vuoto
    if rosa is None:
        raise SystemExit("--probabili ha bisogno anche di --listone: i nomi passano di li'")

    giocatori, _ = lineups.load_probabili(probabili)
    tabelle = [(giocatori, ("slot", "titolarita", "stato"))]
    if piazzati and piazzati.exists():
        tabelle.append((lineups.load_set_pieces(piazzati), ("rigori", "fermo")))

    # dal listone_id all'id sintetico che le righe della rosa portano
    per_listone = {
        str(r.listone_id): (str(r.player_id) if r.player_id else f"L{r.listone_id}")
        for r in rosa.itertuples()
    }
    valori: dict[str, dict[str, object]] = {}
    for tabella, colonne in tabelle:
        agganciata, fuori = lineups.attach(tabella, rosa)
        if fuori:
            print(f"  {len(fuori)} nomi non sono nel listone, ignorati: " + ", ".join(fuori))
        for row in agganciata.itertuples():
            pid = per_listone.get(str(row.listone_id))
            if pid is None:
                continue
            for colonna in colonne:
                valore = getattr(row, colonna)
                if valore is None or (isinstance(valore, float) and pd.isna(valore)):
                    continue
                valori.setdefault(colonna, {})[pid] = valore

    ids = archive["player_id"].astype(str)
    out = vuoto.copy()
    for colonna, mappa in valori.items():
        mappato = ids.map(mappa)
        out[colonna] = mappato.astype(float) if vuoto[colonna].dtype == float else mappato
    return out


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


def _recent_form(archive: pd.DataFrame, season: str, gameweek: int) -> dict[str, list[float]]:
    """Gli ultimi fantavoti presi, dal piu' vecchio al piu' recente.

    Se della stagione in corso non c'e' ancora niente -- il caso della prima
    giornata -- si mostrano gli ultimi della stagione scorsa. E' quello che
    l'utente ha in testa ad agosto, ed e' la stessa informazione da cui parte
    la stima.
    """
    rules = Rules()
    prima = archive[(archive["season"] == season) & (archive["gameweek"] < gameweek)]
    if prima.empty:
        prima = archive[archive["season"] == _previous_label(season)]
    prima = prima[prima["played"]].sort_values("gameweek")
    out: dict[str, list[float]] = {}
    for pid, gruppo in prima.groupby("player_id", observed=True):
        coda = gruppo.tail(FORM_WINDOW)
        out[str(pid)] = [
            round(fantavoto(float(v), r, rules), 1)
            for v, r in zip(coda["voto"], coda[list(EVENTS)].to_dict("records"))
        ]
    return out


def _int(value):
    return None if pd.isna(value) else int(value)


def _round(value, digits: int = 2):
    return None if pd.isna(value) else round(float(value), digits)


def _clean(value):
    return None if pd.isna(value) else str(value)


if __name__ == "__main__":
    main()
