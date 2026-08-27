"""Scarica le quote 1X2 di Serie A da The Odds API.

Produce un CSV compatibile con `fantasquama.fixtures.load_fixtures`, quindi
con le stesse colonne `AvgH`, `AvgD` e `AvgA` che arrivavano da
football-data.co.uk. La chiave si legge da `THE_ODDS_API_KEY`, oppure da un
file `.env` locale ignorato da git.

    python fetch_odds.py 2026
    python fetch_odds.py --base ../ios/FantaSquama/Resources/serieA.json -o odds-current.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fantasquama.fixtures import _canonical


API_HOST = "https://api.the-odds-api.com"
SPORT = "soccer_italy_serie_a"
REGION = "eu"
MARKET = "h2h"
ENV_KEY = "THE_ODDS_API_KEY"
FIXTURES = Path("data/fixtures")
OUTPUT_FIELDS = (
    "Div", "Date", "Time", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA",
    "Bookmakers", "LastUpdate", "Source",
)


@dataclass(frozen=True)
class TargetMatch:
    home: str
    away: str
    commence_time: str
    matchday: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (_canonical(self.home), _canonical(self.away))


@dataclass(frozen=True)
class OddsRow:
    home: str
    away: str
    commence_time: str
    home_odds: float
    draw_odds: float
    away_odds: float
    bookmakers: int
    last_update: str

    @property
    def key(self) -> tuple[str, str]:
        return (_canonical(self.home), _canonical(self.away))

    def csv_row(self) -> dict[str, str]:
        date, time = _format_date_time(self.commence_time)
        return {
            "Div": "I1",
            "Date": date,
            "Time": time,
            "HomeTeam": self.home,
            "AwayTeam": self.away,
            "AvgH": _format_decimal(self.home_odds),
            "AvgD": _format_decimal(self.draw_odds),
            "AvgA": _format_decimal(self.away_odds),
            "Bookmakers": str(self.bookmakers),
            "LastUpdate": self.last_update,
            "Source": "the-odds-api",
        }


def read_api_key() -> str:
    token = os.environ.get(ENV_KEY, "").strip()
    if token:
        return token

    for path in (Path(".env"), Path(__file__).resolve().parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == ENV_KEY:
                token = value.strip().strip('"').strip("'")
                if token:
                    return token
    raise SystemExit(f"manca {ENV_KEY}: esportala nell'ambiente o mettila in backtest/.env")


def fetch_odds(api_key: str, sport: str = SPORT, region: str = REGION) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({
        "apiKey": api_key,
        "regions": region,
        "markets": MARKET,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    })
    url = f"{API_HOST}/v4/sports/{sport}/odds/?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "FantaSquama odds updater"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read())


def load_targets(matches: Path | None, base: Path | None) -> list[TargetMatch]:
    if base is not None:
        payload = json.loads(base.read_text())
        gameweek = payload.get("gameweek")
        return [
            TargetMatch(m["home"], m["away"], m["date"], m.get("matchday"))
            for m in payload.get("matches", [])
            if gameweek is None or m.get("matchday") == gameweek
        ]

    if matches is None or not matches.exists():
        return []

    payload = json.loads(matches.read_text())
    return [
        TargetMatch(
            match["homeTeam"]["shortName"],
            match["awayTeam"]["shortName"],
            match["utcDate"],
            match.get("matchday"),
        )
        for match in payload.get("matches", [])
    ]


def parse_odds(events: list[dict[str, Any]], targets: list[TargetMatch] | None = None) -> list[OddsRow]:
    target_by_key = {target.key: target for target in targets or []}
    rows: list[OddsRow] = []

    for event in events:
        event_home = _canonical(event.get("home_team", ""))
        event_away = _canonical(event.get("away_team", ""))
        target = target_by_key.get((event_home, event_away))
        if targets is not None and target is None:
            continue

        home = target.home if target else event_home
        away = target.away if target else event_away
        averages = _average_h2h(event, event_home, event_away)
        if averages is None:
            continue
        rows.append(OddsRow(
            home=home,
            away=away,
            commence_time=target.commence_time if target else event.get("commence_time", ""),
            home_odds=averages[0],
            draw_odds=averages[1],
            away_odds=averages[2],
            bookmakers=averages[3],
            last_update=_latest_bookmaker_update(event),
        ))

    rows.sort(key=lambda row: row.commence_time)
    return rows


def write_odds(rows: list[OddsRow], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = [row.csv_row() for row in rows]
    if out.exists():
        fresh = _merge_existing(out, fresh)

    fields = _fieldnames(out, fresh)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fresh)


def _average_h2h(event: dict[str, Any], home: str, away: str) -> tuple[float, float, float, int] | None:
    home_prices: list[float] = []
    draw_prices: list[float] = []
    away_prices: list[float] = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != MARKET:
                continue
            prices = {_canonical(outcome.get("name", "")): outcome.get("price") for outcome in market.get("outcomes", [])}
            home_price = _as_float(prices.get(home))
            draw_price = _as_float(prices.get("Draw") or prices.get("Pareggio"))
            away_price = _as_float(prices.get(away))
            if home_price and draw_price and away_price:
                home_prices.append(home_price)
                draw_prices.append(draw_price)
                away_prices.append(away_price)

    if not home_prices:
        return None
    return (
        sum(home_prices) / len(home_prices),
        sum(draw_prices) / len(draw_prices),
        sum(away_prices) / len(away_prices),
        len(home_prices),
    )


def _latest_bookmaker_update(event: dict[str, Any]) -> str:
    updates = [
        str(bookmaker.get("last_update", ""))
        for bookmaker in event.get("bookmakers", [])
        if bookmaker.get("last_update")
    ]
    return max(updates) if updates else datetime.now(timezone.utc).isoformat(timespec="seconds")


def _merge_existing(out: Path, fresh: list[dict[str, str]]) -> list[dict[str, str]]:
    with out.open(newline="") as fh:
        existing = list(csv.DictReader(fh))
    by_match: dict[tuple[str, str], dict[str, str]] = {
        (_canonical(row.get("HomeTeam", "")), _canonical(row.get("AwayTeam", ""))): row
        for row in existing
        if row.get("HomeTeam") and row.get("AwayTeam")
    }
    for row in fresh:
        key = (_canonical(row["HomeTeam"]), _canonical(row["AwayTeam"]))
        by_match[key] = {**by_match.get(key, {}), **row}
    return sorted(by_match.values(), key=lambda row: (_date_sort_key(row), row.get("HomeTeam", "")))


def _fieldnames(out: Path, rows: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    if out.exists():
        with out.open(newline="") as fh:
            fields = list(csv.DictReader(fh).fieldnames or [])
    for field in OUTPUT_FIELDS:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def _date_sort_key(row: dict[str, str]) -> str:
    date = row.get("Date", "")
    time = row.get("Time", "")
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(f"{date} {time or '00:00'}", fmt).isoformat()
        except ValueError:
            pass
    return f"{date} {time}"


def _format_date_time(value: str) -> tuple[str, str]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value[:10], ""
    return dt.strftime("%d/%m/%Y"), dt.strftime("%H:%M")


def _format_decimal(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 1.0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", nargs="?", type=int, help="stagione iniziale: 2026 -> 2026-27")
    parser.add_argument("-o", "--out", type=Path, help="CSV da scrivere")
    parser.add_argument("--matches", type=Path, help="matches_YEAR.json per allineare i nomi locali")
    parser.add_argument("--base", type=Path, help="serieA-base.json/serieA.json per usare la giornata corrente")
    parser.add_argument("--sport", default=SPORT)
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    if args.year is not None and args.matches is None and args.base is None:
        args.matches = FIXTURES / f"matches_{args.year}.json"
    if args.out is None:
        if args.year is None:
            raise SystemExit("specifica -o/--out quando non passi l'anno")
        args.out = FIXTURES / f"odds_{args.year}.csv"

    targets = load_targets(args.matches, args.base)
    events = fetch_odds(read_api_key(), args.sport, args.region)
    rows = parse_odds(events, targets if targets else None)
    write_odds(rows, args.out)

    print(f"{args.out}: {len(rows)} partite con quote da The Odds API")
    if targets and len(rows) < len(targets):
        print(f"  {len(targets) - len(rows)} partite locali non ancora trovate nelle quote")


if __name__ == "__main__":
    main()
