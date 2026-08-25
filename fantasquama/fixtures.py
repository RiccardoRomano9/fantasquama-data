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
    colonne = ["season", "team", "gameweek", "opponent", "home", "p_win", "p_draw", "p_lose"]
    merged = archive[["season", "team", "gameweek"]].merge(
        fixtures[colonne], on=["season", "team", "gameweek"], how="left"
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
