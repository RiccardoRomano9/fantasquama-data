"""Chi giochi contro, in casa o fuori, e quanto e' difficile la partita.

Il modello, senza questo, non sa contro chi si gioca: un attaccante contro il
Lecce e uno contro l'Inter sono identici. E' il buco piu' grande che restava, e
pesa soprattutto sui difensori -- il ruolo dove il modello va peggio.

Due fonti, entrambe gratuite e locali (le scarica `fetch_fixtures.py`, non
questo modulo: qui dentro non si fa rete):

- **football-data.org** dichiara la giornata. E' l'aggancio all'archivio dei
  voti, ed e' il motivo per cui usiamo questa fonte invece di dedurre la
  giornata dalle date: una partita rinviata sposta l'ordine per data e sfasa
  tutte le successive di quella squadra, in silenzio.
- **football-data.co.uk** porta le quote dei bookmaker, che sono di gran lunga
  il miglior indicatore di difficolta' disponibile: fra la fascia piu' sfavorita
  e la piu' favorita i gol fatti variano di 2,7 volte e quelli subiti di 2,8.

Le probabilita' implicite sono normalizzate a somma 1, cioe' tolto il margine
del bookmaker: le quote grezze sommano a piu' di 1 perche' quello e' il suo
guadagno, e usarle senza normalizzare gonfierebbe ogni probabilita'.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# I nomi delle squadre nelle tre fonti. La colonna canonica e' quella
# dell'archivio dei voti, che e' il dato principale.
TEAM_ALIASES: dict[str, str] = {
    # football-data.org (shortName) -> archivio
    "Venezia FC": "Venezia",
    "AC Pisa": "Pisa",
    # football-data.co.uk -> archivio
    "Como": "Como 1907",
}

FIXTURE_COLUMNS: tuple[str, ...] = (
    "season", "gameweek", "team", "opponent", "home",
    "goals_for", "goals_against", "p_win", "p_draw", "p_lose",
)

# Fotografia della squadra calcolata *prima* di quella giornata. Queste
# grandezze entrano nel modello appreso; i fattori sintetici in `team_factors`
# danno al modello a regole un fallback sensato quando le quote non ci sono.
TEAM_CONTEXT_FEATURES: tuple[str, ...] = (
    "team_games_before", "team_points_per_game", "team_goal_difference_per_game",
    "team_goals_for_rate", "team_goals_against_rate", "team_form_points_per_game",
    "team_home_points_per_game", "team_away_points_per_game", "team_rank_before",
    "team_schedule_strength",
    "team_prior_points_per_game", "team_prior_goals_for_rate",
    "team_prior_goals_against_rate", "team_prior_rank",
    "opponent_points_per_game", "opponent_goal_difference_per_game",
    "opponent_goals_for_rate", "opponent_goals_against_rate",
    "opponent_form_points_per_game", "opponent_rank_before",
    "opponent_schedule_strength",
    "opponent_prior_points_per_game", "opponent_prior_goals_for_rate",
    "opponent_prior_goals_against_rate", "opponent_prior_rank",
)


def _team_default(name: str) -> float:
    if "rank" in name:
        return 10.5
    if "games" in name:
        return 0.0
    if "goal_difference" in name:
        return 0.0
    if "goals_for" in name or "goals_against" in name:
        return 1.30
    return 1.35  # punti e forma: media teorica di una gara equilibrata


TEAM_CONTEXT_DEFAULTS: dict[str, float] = {name: _team_default(name) for name in TEAM_CONTEXT_FEATURES}

# Sotto questa quota di accordo fra calendario e archivio la mappatura dei nomi
# o l'abbinamento delle giornate si e' rotto, e proseguire vorrebbe dire
# attribuire a meta' campionato l'avversario sbagliato.
MIN_AGREEMENT: float = 0.95


def _season(year: int) -> str:
    """2023 -> "2023-24", la forma usata dalle cartelle dell'archivio."""
    return f"{year}-{str(year + 1)[2:]}"


def _canonical(name: str) -> str:
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


def _implied_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    """Quote decimali -> probabilita' a somma 1, tolto il margine."""
    inverse = 1.0 / odds
    return inverse.div(inverse.sum(axis=1), axis=0)


def load_fixtures(root: Path) -> pd.DataFrame:
    """Una riga per squadra per partita, con avversario, campo e quote.

    `root` e' la cartella prodotta da `fetch_fixtures.py`. Le quote sono
    facoltative: senza il CSV di una stagione le colonne `p_*` restano NaN e
    il modello ricade sul fattore neutro, invece di fermarsi.
    """
    rows: list[tuple] = []
    for path in sorted(root.glob("matches_*.json")):
        year = int(path.stem.split("_")[1])
        season = _season(year)
        odds = _load_odds(root / f"odds_{year}.csv")

        for match in json.loads(path.read_text())["matches"]:
            home = _canonical(match["homeTeam"]["shortName"])
            away = _canonical(match["awayTeam"]["shortName"])
            score = match["score"]["fullTime"]
            gw = match["matchday"]
            date = match["utcDate"][:10]
            pw, pd_, pl = odds.get((home, away), (np.nan, np.nan, np.nan))
            rows.append((season, gw, home, away, True, score["home"], score["away"], pw, pd_, pl))
            rows.append((season, gw, away, home, False, score["away"], score["home"], pl, pd_, pw))

    if not rows:
        raise ValueError(f"nessun calendario trovato in {root} — esegui fetch_fixtures.py")

    fixtures = pd.DataFrame(rows, columns=list(FIXTURE_COLUMNS))
    duplicated = fixtures.duplicated(subset=["season", "team", "gameweek"], keep=False).to_numpy()
    if duplicated.any():
        esempi = fixtures[duplicated][["season", "team", "gameweek"]].head(5)
        raise ValueError(
            f"{int(duplicated.sum())} righe duplicate su (stagione, squadra, giornata):\n"
            f"{esempi.to_string(index=False)}"
        )
    return fixtures


def _load_odds(path: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Quote medie di mercato per (casa, trasferta). Vuoto se il file manca."""
    if not path.exists():
        return {}
    raw = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["HomeTeam", "AwayTeam"])
    colonne = ("AvgH", "AvgD", "AvgA")
    if not set(colonne) <= set(raw.columns):
        return {}
    probabilities = _implied_probabilities(raw[list(colonne)].apply(pd.to_numeric, errors="coerce"))
    return {
        (_canonical(h), _canonical(a)): (pw, pd_, pl)
        for h, a, pw, pd_, pl in zip(
            raw["HomeTeam"], raw["AwayTeam"],
            probabilities["AvgH"], probabilities["AvgD"], probabilities["AvgA"],
        )
    }


def team_context(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge situazione e forma squadra note prima di ogni partita.

    Il calendario contiene anche le giornate future, ma ogni cumulato esclude
    la riga corrente e conta solo risultati conclusi. In questo modo punti,
    classifica e forma sono utilizzabili nel backtest senza guardare il futuro.
    """
    required = set(FIXTURE_COLUMNS)
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"fixture senza colonne richieste: {sorted(missing)}")

    out = fixtures.copy().sort_values(["season", "team", "gameweek"])
    keys = [out["season"], out["team"]]
    gf = pd.to_numeric(out["goals_for"], errors="coerce")
    ga = pd.to_numeric(out["goals_against"], errors="coerce")
    finished = gf.notna() & ga.notna()
    points = pd.Series(np.select([gf > ga, gf == ga], [3.0, 1.0], default=0.0), index=out.index)
    points = points.where(finished)

    def before(values: pd.Series) -> pd.Series:
        """Cumulato stagionale esclusa la giornata della riga."""
        known = values.fillna(0.0)
        return known.groupby(keys).cumsum() - known

    games = before(finished.astype(float))
    points_before = before(points)
    gf_before = before(gf.where(finished))
    ga_before = before(ga.where(finished))
    gd_before = gf_before - ga_before

    out["team_games_before"] = games
    out["team_points_per_game"] = points_before / games.replace(0.0, np.nan)
    out["team_goal_difference_per_game"] = gd_before / games.replace(0.0, np.nan)
    out["team_goals_for_rate"] = gf_before / games.replace(0.0, np.nan)
    out["team_goals_against_rate"] = ga_before / games.replace(0.0, np.nan)

    # Forma su cinque risultati effettivamente conclusi, sempre sfalsata.
    out["team_form_points_per_game"] = points.groupby(keys, group_keys=False).transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )

    home = out["home"].fillna(False).astype(bool)
    for label, mask in (("home", home), ("away", ~home)):
        local_games = before((finished & mask).astype(float))
        local_points = before(points.where(mask))
        out[f"team_{label}_points_per_game"] = local_points / local_games.replace(0.0, np.nan)

    # Posizione ad apertura della giornata: punti, differenza reti, gol fatti.
    standings = out[["season", "gameweek", "team"]].copy()
    standings["points"] = points_before.to_numpy()
    standings["difference"] = gd_before.to_numpy()
    standings["goals_for"] = gf_before.to_numpy()
    standings = standings.sort_values(
        ["season", "gameweek", "points", "difference", "goals_for", "team"],
        ascending=[True, True, False, False, False, True],
    )
    standings["team_rank_before"] = standings.groupby(["season", "gameweek"]).cumcount() + 1
    out = out.merge(
        standings[["season", "gameweek", "team", "team_rank_before"]],
        on=["season", "gameweek", "team"], how="left", validate="one_to_one",
    )
    out.loc[out["team_games_before"] == 0, "team_rank_before"] = np.nan

    # Difficoltà del cammino già percorso: per ogni gara passata conserva la
    # forza che l'avversaria aveva *prima* di quel fischio d'inizio, poi ne fa
    # la media. Non usa la sua classifica maturata dopo averla affrontata.
    opponent_snapshot = out[["season", "gameweek", "team", "team_points_per_game"]].rename(columns={
        "team": "opponent", "team_points_per_game": "_opponent_points_at_match",
    })
    out = out.merge(opponent_snapshot, on=["season", "gameweek", "opponent"], how="left", validate="many_to_one")
    opponent_strength = out["_opponent_points_at_match"].fillna(1.35)
    schedule_keys = [out["season"], out["team"]]
    out["team_schedule_strength"] = opponent_strength.groupby(schedule_keys, group_keys=False).transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )
    out = out.drop(columns="_opponent_points_at_match")

    out = _attach_previous_season_strength(out, finished, points, gf, ga)
    return _attach_opponent_context(out)


def _attach_previous_season_strength(
    out: pd.DataFrame, finished: pd.Series, points: pd.Series, gf: pd.Series, ga: pd.Series
) -> pd.DataFrame:
    source = out.assign(
        _finished=finished.to_numpy(float), _points=points.fillna(0.0).to_numpy(),
        _gf=gf.where(finished).fillna(0.0).to_numpy(), _ga=ga.where(finished).fillna(0.0).to_numpy(),
    )
    summary = source.groupby(["season", "team"], as_index=False, observed=True).agg(
        games=("_finished", "sum"), points=("_points", "sum"),
        goals_for=("_gf", "sum"), goals_against=("_ga", "sum"),
    )
    summary["difference"] = summary["goals_for"] - summary["goals_against"]
    summary = summary.sort_values(
        ["season", "points", "difference", "goals_for", "team"],
        ascending=[True, False, False, False, True],
    )
    summary["rank"] = summary.groupby("season").cumcount() + 1
    summary["season"] = summary["season"].map(_next_season)
    summary["prior_points_per_game"] = summary["points"] / summary["games"].replace(0.0, np.nan)
    summary["prior_goals_for_rate"] = summary["goals_for"] / summary["games"].replace(0.0, np.nan)
    summary["prior_goals_against_rate"] = summary["goals_against"] / summary["games"].replace(0.0, np.nan)
    summary = summary.rename(columns={"rank": "prior_rank"})
    return out.merge(
        summary[["season", "team", "prior_points_per_game", "prior_goals_for_rate", "prior_goals_against_rate", "prior_rank"]],
        on=["season", "team"], how="left", validate="many_to_one",
    ).rename(columns={
        "prior_points_per_game": "team_prior_points_per_game",
        "prior_goals_for_rate": "team_prior_goals_for_rate",
        "prior_goals_against_rate": "team_prior_goals_against_rate",
        "prior_rank": "team_prior_rank",
    })


def _next_season(label: str) -> str:
    year = int(str(label)[:4]) + 1
    return f"{year}-{str(year + 1)[2:]}"


def _attach_opponent_context(out: pd.DataFrame) -> pd.DataFrame:
    own = [name for name in TEAM_CONTEXT_FEATURES if name.startswith("team_")]
    other = out[["season", "gameweek", "team", *own]].rename(columns={
        "team": "opponent",
        **{name: name.replace("team_", "opponent_", 1) for name in own},
    })
    return out.merge(other, on=["season", "gameweek", "opponent"], how="left", validate="many_to_one")


def team_factors(context: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fattori squadra per il modello a regole, utili senza quote di mercato.

    Il presente pesa gradualmente (al massimo 70%) e il passato stagione
    precedente fa da ancora. Così un 3-0 della prima giornata non trasforma da
    solo una squadra in un super-attacco né un pareggio in una difesa perfetta.
    """
    def values(name: str) -> np.ndarray:
        if name not in context:
            return np.full(len(context), np.nan)
        return pd.to_numeric(context[name], errors="coerce").to_numpy(np.float64)

    def blended_rate(now: str, prior: str, games: str, default: float) -> np.ndarray:
        current = values(now)
        old = values(prior)
        count = values(games)
        weight = np.clip(np.nan_to_num(count, nan=0.0) / 8.0, 0.0, 0.70)
        anchor = np.where(np.isfinite(old), old, default)
        return np.where(np.isfinite(current), weight * current + (1.0 - weight) * anchor, anchor)

    own_attack = blended_rate("team_goals_for_rate", "team_prior_goals_for_rate", "team_games_before", 1.30)
    opponent_defense = blended_rate(
        "opponent_goals_against_rate", "opponent_prior_goals_against_rate", "opponent_games_before", 1.30
    )
    own_defense = blended_rate("team_goals_against_rate", "team_prior_goals_against_rate", "team_games_before", 1.30)
    opponent_attack = blended_rate(
        "opponent_goals_for_rate", "opponent_prior_goals_for_rate", "opponent_games_before", 1.30
    )

    own_points = blended_rate("team_points_per_game", "team_prior_points_per_game", "team_games_before", 1.35)
    opponent_points = blended_rate(
        "opponent_points_per_game", "opponent_prior_points_per_game", "opponent_games_before", 1.35
    )
    # Gol sono il segnale principale; rendimento in punti aggiunge una piccola
    # correzione di solidità senza contare due volte la differenza reti.
    attack = np.sqrt((own_attack / 1.30) * (opponent_defense / 1.30))
    attack *= np.clip((own_points / 1.35) ** 0.12, 0.90, 1.10)
    defense = np.sqrt((own_defense / 1.30) * (opponent_attack / 1.30))
    defense *= np.clip((opponent_points / 1.35) ** 0.10, 0.90, 1.10)
    return np.clip(attack, 0.65, 1.55), np.clip(defense, 0.65, 1.55)


def agreement(fixtures: pd.DataFrame, archive: pd.DataFrame) -> float:
    """Quota di righe dove i gol subiti dell'archivio confermano il calendario.

    E' il controllo che dice se la mappatura dei nomi e l'abbinamento delle
    giornate reggono. I gol *subiti* sono il termine di paragone giusto perche'
    nell'archivio sono esatti: includono gli autogol, che invece mancano dalla
    somma dei gol fatti dai giocatori.
    """
    subiti = (
        archive.groupby(["season", "team", "gameweek"], as_index=False, observed=True)
        .agg(archive_against=("gs", "sum"))
    )
    merged = subiti.merge(fixtures, on=["season", "team", "gameweek"], how="inner")
    # Una partita non ancora giocata non ha un risultato con cui confermare
    # niente. Contarla come disaccordo farebbe fallire il controllo proprio
    # sulla stagione in corso -- l'unica che all'app interessa davvero.
    merged = merged[merged["goals_against"].notna()]
    if merged.empty:
        return 0.0
    return float((merged["archive_against"] == merged["goals_against"]).mean())


def attach(archive: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Aggancia il calendario alle righe dell'archivio, verificando prima.

    Ritorna un frame con lo stesso indice e lo stesso ordine di `archive`.
    """
    quota = agreement(fixtures, archive)
    if quota < MIN_AGREEMENT:
        raise ValueError(
            f"calendario e archivio concordano solo sul {quota:.1%} delle righe "
            f"(minimo {MIN_AGREEMENT:.0%}). Controllare TEAM_ALIASES: attribuire "
            f"l'avversario sbagliato falserebbe ogni consiglio senza dare segnale."
        )
    enriched = team_context(fixtures)
    colonne = ["season", "team", "gameweek", "opponent", "home", "p_win", "p_draw", "p_lose", *TEAM_CONTEXT_FEATURES]
    merged = archive[["season", "team", "gameweek"]].merge(
        enriched[colonne], on=["season", "team", "gameweek"], how="left"
    )
    # il merge azzera l'indice: lo rimettiamo posizionalmente, come ovunque
    merged.index = archive.index
    return merged.drop(columns=["season", "team", "gameweek"])


# Quanto la difficolta' della partita puo' spostare la stima. Piu' larghi dei
# limiti sul contesto squadra perche' il segnale e' piu' forte: fra la fascia
# piu' sfavorita e la piu' favorita i gol fatti variano di 2,7 volte.
#
# ponytail: estremi a occhio, come K. Da tarare sul backtest.
MATCH_FACTOR_MIN: float = 0.4
MATCH_FACTOR_MAX: float = 2.5


@dataclass(frozen=True)
class Difficulty:
    """Da quanto una squadra e' favorita a quanto ci si aspetta segni e subisca.

    Due rette sul vantaggio di mercato (`p_win - p_lose`), tarate sulle sole
    stagioni di taratura. I coefficienti sono i gol attesi; i fattori sono quei
    gol rapportati alla media, cosi' una partita di media difficolta' vale 1.
    """

    attack: tuple[float, float]
    defense: tuple[float, float]
    mean_attack: float
    mean_defense: float

    def factors(self, advantage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fattore offensivo e difensivo per riga. NaN -> 1, cioe' nessuna correzione."""
        return (
            _factor(advantage, self.attack, self.mean_attack),
            _factor(advantage, self.defense, self.mean_defense),
        )


def _factor(advantage: np.ndarray, retta: tuple[float, float], media: float) -> np.ndarray:
    pendenza, intercetta = retta
    atteso = pendenza * advantage + intercetta
    factor = atteso / media
    factor[~np.isfinite(factor)] = 1.0
    return np.clip(factor, MATCH_FACTOR_MIN, MATCH_FACTOR_MAX)


def fit_difficulty(fixtures: pd.DataFrame, seasons: list[str]) -> Difficulty:
    """Tara le due rette sulle stagioni date.

    Solo quelle: usare anche la stagione di verifica farebbe entrare il suo
    futuro nel calcolo di una grandezza usata per predirla.
    """
    rows = fixtures[fixtures["season"].isin(seasons)]
    advantage = (rows["p_win"] - rows["p_lose"]).to_numpy(np.float64)
    gf = pd.to_numeric(rows["goals_for"], errors="coerce").to_numpy(np.float64)
    ga = pd.to_numeric(rows["goals_against"], errors="coerce").to_numpy(np.float64)

    usable = np.isfinite(advantage) & np.isfinite(gf) & np.isfinite(ga)
    if usable.sum() < 100:
        raise ValueError(
            f"solo {int(usable.sum())} partite con quote nelle stagioni {seasons}: "
            f"troppo poche per tarare la difficolta'."
        )

    attack = tuple(np.polyfit(advantage[usable], gf[usable], 1))
    defense = tuple(np.polyfit(advantage[usable], ga[usable], 1))
    return Difficulty(
        attack=attack, defense=defense,
        mean_attack=float(gf[usable].mean()), mean_defense=float(ga[usable].mean()),
    )
