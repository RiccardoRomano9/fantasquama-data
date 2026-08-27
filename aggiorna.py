"""Applica le probabili formazioni a `serieA-base.json` e scrive `serieA.json`.

E' il pezzo che gira ogni due ore su GitHub Actions. Non tocca l'archivio dei
voti e non ne ha bisogno: la parte lenta -- le stime del modello -- e' gia'
dentro al file base, prodotto a mano quando arrivano voti nuovi. Qui cambia
solo cio' che cambia ogni settimana: chi gioca, chi e' in panchina, chi e'
fuori.

    python aggiorna.py                      # scarica e aggiorna in loco
    python aggiorna.py --probabili p.json   # usa un file gia' scaricato

**Non c'e' una seconda implementazione della formula.** Le costanti e le
funzioni sono quelle di `fantasquama.lineups`, copiate qui da `build.py`
insieme al resto: se cambiano di la', si rigenera. Riscriverle a mano sarebbe
il modo piu' sicuro di farle divergere proprio mentre nessuno guarda.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fantasquama import fixtures
from fantasquama import lineups
from fantasquama import roster
from fantasquama.scoring import EVENTS

ODDS_STALE_HOURS = 24
TEAM_SCALED_EVENTS = ("gf", "rf", "ass")
PROP_ODDS_MARKETS = (
    "player_goal_scorer_anytime",
    "player_assists",
    "player_to_receive_card",
    "btts",
)
PROP_ODDS_REGION = "us"
PROP_MONTHLY_CREDIT_CAP = 420
PROP_EARLY_MIN_HOURS = 48
PROP_EARLY_MAX_HOURS = 96
PROP_DAY_BEFORE_MAX_HOURS = 30


def scarica(
    script: str, destinazione: Path, obbligatorio: bool = True, extra: list[str] | None = None
) -> Path | None:
    """Esegue uno degli `fetch_*` che stanno accanto a questo file."""
    percorso = Path(__file__).resolve().parent / script
    esito = subprocess.run([sys.executable, str(percorso), "-o", str(destinazione), *(extra or [])])
    if esito.returncode != 0:
        if obbligatorio:
            raise SystemExit(f"{script} non e' riuscito: nessun aggiornamento")
        print(f"  {script} non e' riuscito: si tiene quello che c'era")
        return None
    return destinazione


def aggiorna(
    base: dict,
    probabili: Path,
    quote: Path | None = None,
    prop_quote: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Il file base con la titolarita' rifatta secondo le probabili."""
    now = now or datetime.now(timezone.utc)
    giocatori = base["players"]
    rosa = pd.DataFrame({
        "listone_id": [p["id"] for p in giocatori],
        "player_name": [p["name"] for p in giocatori],
        "team": [p["team"] for p in giocatori],
        "role": [p["role"] for p in giocatori],
    })

    formazione, squadre = lineups.load_probabili(probabili)
    agganciata, persi = lineups.attach(formazione, rosa)
    if persi:
        print(f"  {len(persi)} nomi non sono in rosa, ignorati: {', '.join(persi)}")
    squadre = lineups.enrich_ballottaggi(squadre, rosa)

    per_id = {r.listone_id: r for r in agganciata.itertuples() if r.listone_id}
    slot = pd.Series([per_id[p["id"]].slot if p["id"] in per_id else None
                      for p in giocatori], dtype=object)
    pct = pd.Series([float(per_id[p["id"]].titolarita) if p["id"] in per_id else float("nan")
                     for p in giocatori], dtype=float)
    rank = pd.Series([p.get("penaltyRank") for p in giocatori], dtype=object)

    probabilita = pd.DataFrame({
        **{evento: [float(p["events"][evento]) for p in giocatori] for evento in EVENTS},
        "p_vote": [float(p["playProbability"]) for p in giocatori],
    })
    aggiornate = lineups.apply(
        probabilita,
        pd.Series([p["role"] for p in giocatori]),
        pd.Series([p["team"] for p in giocatori]),
        slot, pct, rank,
    )

    for i, giocatore in enumerate(giocatori):
        riga = per_id.get(giocatore["id"])
        giocatore["playProbability"] = round(float(aggiornate["p_vote"].iloc[i]), 3)
        for evento in ("rf", "rs"):
            giocatore["events"][evento] = round(float(aggiornate[evento].iloc[i]), 4)
        giocatore["lineupSlot"] = riga.slot if riga is not None else None
        giocatore["startingProbability"] = float(riga.titolarita) if riga is not None else None
        giocatore["status"] = (riga.stato or None) if riga is not None else None
        # La stima imparata non conosce le formazioni: se restasse quella di
        # prima, l'app ne farebbe la media con una che sa che il giocatore e'
        # infortunato, e il risultato non sarebbe ne' l'una ne' l'altra.
        giocatore["learnedPlayProbability"] = giocatore["playProbability"]

    aliases = lineups.roster.TEAM_ALIASES
    base["teams"] = [
        {
            "name": aliases.get(str(r.squadra), str(r.squadra)),
            "formation": str(r.modulo),
            "isOfficial": bool(r.ufficiale),
            "ballottaggi": r.ballottaggi,
        }
        for r in squadre.itertuples()
    ]
    quote_note = ""
    if quote is not None:
        aggiornate = aggiorna_quote(base, quote)
        if aggiornate:
            base["oddsUpdatedAt"] = now.isoformat(timespec="seconds")
            base["oddsSource"] = "the-odds-api"
            quote_note = f" Probabilita' partita aggiornate con le quote 1X2 ({aggiornate} partite)."
    if prop_quote is not None:
        aggiorna_prop_quote(base, prop_quote)
    contestualizzati = applica_contesto_quote(base)
    if contestualizzati:
        print(f"  quote applicate alle probabilita' evento di {contestualizzati} giocatori")
    prop_contestualizzati = applica_prop_quote(base)
    if prop_contestualizzati:
        print(f"  props bookmaker applicati a {prop_contestualizzati} giocatori")
    base["generatedAt"] = now.isoformat(timespec="seconds")
    base["note"] = (
        base.get("baseNote", base.get("note", ""))
        + " Titolarita', panchina, ballottaggi e indisponibili aggiornati dalle probabili "
        + "formazioni del momento."
        + quote_note
    )
    return base


def riusa_quote(base: dict, precedente: dict | None) -> int:
    """Porta avanti le quote gia' scaricate per la stessa giornata."""
    if not precedente:
        return 0
    if base.get("season") != precedente.get("season") or base.get("gameweek") != precedente.get("gameweek"):
        return 0
    if _base_has_fresher_odds(base, precedente):
        _riusa_prop_quote(base, precedente)
        return 0

    per_id = {
        player.get("id"): player
        for player in precedente.get("players", [])
        if player.get("id") is not None
    }
    riusate = 0
    for player in base["players"]:
        old = per_id.get(player.get("id"))
        if not old:
            continue
        stessa_partita = (
            player.get("team") == old.get("team")
            and player.get("opponent") == old.get("opponent")
            and player.get("home") == old.get("home")
        )
        if not stessa_partita or old.get("winProbability") is None:
            continue
        player["winProbability"] = old.get("winProbability")
        player["drawProbability"] = old.get("drawProbability")
        riusate += 1

    if riusate:
        base["oddsUpdatedAt"] = precedente.get("oddsUpdatedAt") or precedente.get("generatedAt")
        base["oddsSource"] = precedente.get("oddsSource", "the-odds-api")
        _riusa_prop_quote(base, precedente)
    return riusate


def _riusa_prop_quote(base: dict, precedente: dict) -> None:
    for campo in (
        "propOdds", "propOddsUpdatedAt", "propOddsSource", "propOddsCheckedAt",
        "propOddsStages", "propOddsUsage",
    ):
        if campo in precedente:
            base[campo] = precedente[campo]


def _base_has_fresher_odds(base: dict, precedente: dict) -> bool:
    if not any(player.get("winProbability") is not None for player in base.get("players", [])):
        return False
    base_updated = _parse_time(base.get("oddsUpdatedAt"))
    previous_updated = _parse_time(precedente.get("oddsUpdatedAt") or precedente.get("generatedAt"))
    if base_updated is None:
        return False
    return previous_updated is None or base_updated >= previous_updated


def aggiorna_quote(base: dict, quote: Path) -> int:
    """Aggiorna le probabilita' partita dei giocatori dal CSV quote."""
    odds = fixtures._load_odds(quote)
    updated_matches: set[tuple[str, str]] = set()

    for player in base["players"]:
        team = fixtures._canonical(player.get("team", ""))
        opponent = fixtures._canonical(player.get("opponent", ""))
        if not opponent:
            continue
        if player.get("home") is True:
            key = (team, opponent)
            values = odds.get(key)
            win_index = 0
        else:
            key = (opponent, team)
            values = odds.get(key)
            win_index = 2
        if values is None:
            continue
        player["winProbability"] = round(float(values[win_index]), 3)
        player["drawProbability"] = round(float(values[1]), 3)
        updated_matches.add(key)
    return len(updated_matches)


def aggiorna_prop_quote(base: dict, quote: Path) -> None:
    """Aggancia al payload la cache compatta dei props."""
    data = json.loads(quote.read_text())
    base["propOdds"] = data
    base["propOddsUpdatedAt"] = data.get("updatedAt")
    base["propOddsSource"] = data.get("source", "the-odds-api")


def applica_contesto_quote(base: dict) -> int:
    """Scala eventi e gol subiti con le quote nuove della giornata."""
    difficulty = base.get("marketDifficulty")
    if not difficulty:
        return 0

    updated = 0
    for player in base["players"]:
        p_win = _as_float(player.get("winProbability"))
        p_draw = _as_float(player.get("drawProbability"))
        context = player.get("matchContext") or {}
        base_attack = _as_float(context.get("attack"))
        base_defense = _as_float(context.get("defense"))
        if p_win is None or p_draw is None or base_attack is None or base_defense is None:
            continue
        p_lose = max(0.0, 1.0 - p_win - p_draw)
        advantage = p_win - p_lose
        market_attack = _difficulty_factor(difficulty, "attack", advantage)
        market_defense = _difficulty_factor(difficulty, "defense", advantage)

        if context.get("hadMarket"):
            previous_market_attack = _as_float(context.get("marketAttack")) or 1.0
            previous_market_defense = _as_float(context.get("marketDefense")) or 1.0
            new_attack = base_attack * market_attack / previous_market_attack
            new_defense = base_defense * market_defense / previous_market_defense
        else:
            new_attack = market_attack * (base_attack ** 0.20)
            new_defense = market_defense * (base_defense ** 0.20)

        _scale_player_events(player, "events", new_attack / base_attack, new_defense / base_defense)
        _scale_player_events(player, "learnedEvents", new_attack / base_attack, new_defense / base_defense)
        updated += 1
    return updated


def applica_prop_quote(base: dict) -> int:
    """Usa player props e BTTS per rifinire eventi individuali."""
    data = base.get("propOdds")
    if not isinstance(data, dict):
        return 0

    per_giocatore, btts_per_partita = _prop_signals(base, data)
    updated = 0
    for player in base.get("players", []):
        signals = per_giocatore.get(str(player.get("id")))
        btts = btts_per_partita.get(_player_match_key(player))
        changed = False
        if signals:
            player["marketProps"] = signals
            changed = _apply_player_props(player, signals)
        if btts is not None and str(player.get("role")) == "P":
            changed = _apply_btts(player, btts) or changed
        if changed:
            updated += 1
    return updated


def _prop_signals(base: dict, data: dict) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    players = list(base.get("players", []))
    by_match: dict[tuple[str, str], list[dict]] = {}
    for player in players:
        key = _player_match_key(player)
        if key[0] and key[1]:
            by_match.setdefault(key, []).append(player)

    per_giocatore: dict[str, dict[str, float]] = {}
    btts: dict[tuple[str, str], float] = {}
    for match in data.get("matches", []):
        key = (fixtures._canonical(match.get("home", "")), fixtures._canonical(match.get("away", "")))
        markets = match.get("markets") or {}
        if isinstance(markets.get("btts"), dict):
            yes = _as_probability(markets["btts"].get("yes"))
            if yes is not None:
                btts[key] = yes
        index = _player_index(by_match.get(key, []))
        for market in PROP_ODDS_MARKETS:
            if market == "btts":
                continue
            rows = markets.get(market) or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                player = _match_prop_player(row.get("player"), index)
                probability = _as_probability(row.get("probability"))
                if player is None or probability is None:
                    continue
                per_giocatore.setdefault(str(player.get("id")), {})[market] = probability
    return per_giocatore, btts


def _player_index(players: list[dict]) -> dict[tuple[tuple[str, ...], str], list[dict]]:
    out: dict[tuple[tuple[str, ...], str], list[dict]] = {}
    for player in players:
        for value in (player.get("fullName"), player.get("name")):
            tokens = roster.name_tokens(value)
            if not tokens:
                continue
            key = (tokens, roster.name_initial(value))
            out.setdefault(key, []).append(player)
            out.setdefault((tokens, ""), []).append(player)
    return out


def _match_prop_player(name: object, index: dict[tuple[tuple[str, ...], str], list[dict]]) -> dict | None:
    tokens = roster.name_tokens(name)
    if not tokens:
        return None
    initial = roster.name_initial(name)
    candidates = index.get((tokens, initial), [])
    if len(candidates) != 1:
        candidates = index.get((tokens, ""), [])
    if len(candidates) == 1:
        return candidates[0]
    wider = [
        player for (other, _), players in index.items()
        if set(tokens) <= set(other) or set(other) <= set(tokens)
        for player in players
    ]
    unique = {str(player.get("id")): player for player in wider}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _player_match_key(player: dict) -> tuple[str, str]:
    team = fixtures._canonical(player.get("team", ""))
    opponent = fixtures._canonical(player.get("opponent", ""))
    if player.get("home") is True:
        return team, opponent
    return opponent, team


def _apply_player_props(player: dict, signals: dict[str, float]) -> bool:
    changed = False
    for key in ("events", "learnedEvents"):
        events = player.get(key)
        if not isinstance(events, dict):
            continue
        if signals.get("player_goal_scorer_anytime") is not None:
            ratio = _prop_ratio(
                float(events.get("gf") or 0.0) + float(events.get("rf") or 0.0),
                signals["player_goal_scorer_anytime"],
                weight=0.45,
                floor=0.55,
                ceiling=1.75,
            )
            for event in ("gf", "rf"):
                if events.get(event) is not None:
                    events[event] = round(float(events[event]) * ratio, 4)
                    changed = True
        if signals.get("player_assists") is not None and events.get("ass") is not None:
            ratio = _prop_ratio(float(events["ass"]), signals["player_assists"], 0.40, 0.60, 1.65)
            events["ass"] = round(float(events["ass"]) * ratio, 4)
            changed = True
        if signals.get("player_to_receive_card") is not None and events.get("amm") is not None:
            ratio = _prop_ratio(float(events["amm"]), signals["player_to_receive_card"], 0.35, 0.70, 1.45)
            events["amm"] = round(float(events["amm"]) * ratio, 4)
            changed = True
    return changed


def _apply_btts(player: dict, probability: float) -> bool:
    changed = False
    gs_ratio = min(max(1.0 + (probability - 0.50) * 0.60, 0.80), 1.20)
    cs_ratio = min(max(1.0 - (probability - 0.50) * 0.80, 0.75), 1.25)
    for key in ("events", "learnedEvents"):
        events = player.get(key)
        if not isinstance(events, dict):
            continue
        if events.get("gs") is not None:
            events["gs"] = round(float(events["gs"]) * gs_ratio, 4)
            changed = True
        if events.get("cs") is not None:
            events["cs"] = round(float(events["cs"]) * cs_ratio, 4)
            changed = True
    return changed


def _prop_ratio(current: float, target: float, weight: float, floor: float, ceiling: float) -> float:
    if current <= 0 or target <= 0:
        return 1.0
    return min(max((target / current) ** weight, floor), ceiling)


def _scale_player_events(player: dict, key: str, attack_ratio: float, defense_ratio: float) -> None:
    events = player.get(key)
    if not isinstance(events, dict):
        return
    for event in TEAM_SCALED_EVENTS:
        if events.get(event) is not None:
            events[event] = round(float(events[event]) * attack_ratio, 4)
    if events.get("gs") is not None:
        events["gs"] = round(float(events["gs"]) * defense_ratio, 4)


def _difficulty_factor(difficulty: dict, side: str, advantage: float) -> float:
    params = difficulty[side]
    expected = float(params["slope"]) * advantage + float(params["intercept"])
    factor = expected / float(params["mean"])
    limits = difficulty.get("limits", {})
    return min(max(factor, float(limits.get("min", fixtures.MATCH_FACTOR_MIN))), float(limits.get("max", fixtures.MATCH_FACTOR_MAX)))


def _as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number > 0 else None


def _as_probability(value: object) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return min(max(number, 0.01), 0.99)


def deve_scaricare_quote(base: dict, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    if not base.get("oddsUpdatedAt"):
        return True
    if not any(p.get("winProbability") is not None for p in base.get("players", [])):
        return True

    updated = _parse_time(base.get("oddsUpdatedAt"))
    if updated is None:
        return True
    hours = (now - updated).total_seconds() / 3600
    return hours >= _odds_stale_after_hours(base, now)


def _odds_stale_after_hours(base: dict, now: datetime) -> int:
    matches = [m for m in base.get("matches", []) if m.get("matchday") == base.get("gameweek")]
    upcoming = [
        date for date in (_parse_time(m.get("date")) for m in matches)
        if date is not None and date >= now
    ]
    if not upcoming:
        return sys.maxsize
    return ODDS_STALE_HOURS


def deve_scaricare_prop_quote(base: dict, now: datetime | None = None) -> tuple[bool, str | None, str]:
    now = now or datetime.now(timezone.utc)
    stage = _prop_refresh_stage(base, now)
    if stage is None:
        return False, None, "fuori dalla finestra props"
    stages = base.get("propOddsStages") or {}
    if stages.get(stage):
        return False, stage, f"props gia' controllati nella finestra {stage}"
    estimate = _prop_credit_estimate(base, now)
    month = now.strftime("%Y-%m")
    usage = base.get("propOddsUsage") or {}
    used = int(usage.get("estimatedCredits", 0)) if usage.get("month") == month else 0
    if used + estimate > PROP_MONTHLY_CREDIT_CAP:
        return False, stage, (
            f"cap props: {used}+{estimate}>{PROP_MONTHLY_CREDIT_CAP} crediti stimati"
        )
    return True, stage, f"finestra props {stage}"


def registra_tentativo_prop_quote(base: dict, stage: str | None, now: datetime, charged: bool) -> None:
    if stage is None:
        return
    base.setdefault("propOddsStages", {})[stage] = now.isoformat(timespec="seconds")
    base["propOddsCheckedAt"] = now.isoformat(timespec="seconds")
    if charged:
        estimate = _prop_credit_estimate(base, now)
        month = now.strftime("%Y-%m")
        usage = base.get("propOddsUsage") or {}
        previous = int(usage.get("estimatedCredits", 0)) if usage.get("month") == month else 0
        base["propOddsUsage"] = {"month": month, "estimatedCredits": previous + estimate}


def _prop_refresh_stage(base: dict, now: datetime) -> str | None:
    kickoff = _first_upcoming_match(base, now)
    if kickoff is None:
        return None
    hours = (kickoff - now).total_seconds() / 3600
    if PROP_EARLY_MIN_HOURS <= hours <= PROP_EARLY_MAX_HOURS:
        return "early"
    if 0 <= hours <= PROP_DAY_BEFORE_MAX_HOURS:
        return "day-before"
    return None


def _first_upcoming_match(base: dict, now: datetime) -> datetime | None:
    matches = [m for m in base.get("matches", []) if m.get("matchday") == base.get("gameweek")]
    dates = [
        date for date in (_parse_time(m.get("date")) for m in matches)
        if date is not None and date >= now
    ]
    return min(dates) if dates else None


def _prop_credit_estimate(base: dict, now: datetime) -> int:
    upcoming = [
        match for match in base.get("matches", [])
        if match.get("matchday") == base.get("gameweek")
        and (_parse_time(match.get("date")) or now) >= now
    ]
    return len(upcoming) * len(PROP_ODDS_MARKETS)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> None:
    qui = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=qui / "serieA-base.json")
    parser.add_argument("--probabili", type=Path, help="un file gia' scaricato")
    parser.add_argument("--odds", type=Path, help="un CSV quote gia' scaricato")
    parser.add_argument("--prop-odds", type=Path, help="un JSON props gia' scaricato")
    parser.add_argument("--out", type=Path, default=qui / "serieA.json")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    probabili = args.probabili or scarica("fetch_lineups.py", qui / "probabili.json")
    # Le notizie non sono obbligatorie: un giornale che non risponde non deve
    # poter impedire di aggiornare chi gioca, che e' il motivo per cui l'app
    # esiste. Se saltano, restano quelle del giro precedente.
    notizie = scarica("fetch_news.py", qui / "news.json", obbligatorio=False)

    base = json.loads(args.base.read_text())
    base.setdefault("baseNote", base.get("note", ""))
    precedente = json.loads(args.out.read_text()) if args.out.exists() else None
    riusate = riusa_quote(base, precedente)
    if riusate:
        print(f"  quote precedenti riusate per {riusate} giocatori")
    quote = args.odds
    if quote is None:
        if deve_scaricare_quote(base, now):
            quote = scarica(
                "fetch_odds.py", qui / "odds-current.csv", obbligatorio=False,
                extra=["--base", str(args.base)],
            )
        else:
            print(f"  quote ancora fresche ({base.get('oddsUpdatedAt')}): niente chiamata a The Odds API")

    prop_quote = args.prop_odds
    prop_stage = None
    if prop_quote is None:
        scarica_props, prop_stage, motivo_props = deve_scaricare_prop_quote(base, now)
        if scarica_props:
            prop_quote = scarica(
                "fetch_odds.py", qui / "prop-odds-current.json", obbligatorio=False,
                extra=[
                    "--base", str(args.base),
                    "--props-only",
                    "--props-out", str(qui / "prop-odds-current.json"),
                    "--props-region", PROP_ODDS_REGION,
                    "--props-markets", ",".join(PROP_ODDS_MARKETS),
                ],
            )
            registra_tentativo_prop_quote(base, prop_stage, now, charged=prop_quote is not None)
        else:
            print(f"  {motivo_props}: niente player props")
    elif prop_quote.exists():
        prop_stage = _prop_refresh_stage(base, now) or "manual"
        registra_tentativo_prop_quote(base, prop_stage, now, charged=prop_stage != "manual")

    aggiornato = aggiorna(base, probabili, quote, prop_quote, now)
    if notizie is not None:
        aggiornato["news"] = json.loads(notizie.read_text())
    args.out.write_text(json.dumps(aggiornato, ensure_ascii=False, indent=1))
    print(f"{args.out}: {len(aggiornato['players'])} giocatori, "
          f"{len(aggiornato.get('news', []))} notizie, {args.out.stat().st_size:,} byte")


if __name__ == "__main__":
    main()
