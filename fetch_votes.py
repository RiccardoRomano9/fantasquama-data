#!/usr/bin/env python3
"""Scarica i voti Excel (Nuovo) disponibili da Fanta.Soccer.

    python fetch_votes.py --season 2026-2027 --out data

La pagina non ha URL Excel stabili: ogni pulsante invia un postback ASP.NET.
Lo script legge i campi nascosti della pagina, individua le giornate abilitate
e scarica soltanto quelle che non sono gia' nell'archivio locale. Cosi' e'
sicuro eseguirlo ogni giorno da GitHub Actions: quando non c'e' un voto nuovo
esce senza modificare nulla.
"""

import argparse
import html
import re
import sys
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener


URL = "https://www.fanta.soccer/it/archiviovoti/A/{season}/"
UA = "FantaSquama/1.0 (+https://github.com/RiccardoRomano9/fantasquama-data)"
POSTBACK = re.compile(r"__doPostBack\('([^']+)',\s*'([^']*)'\)")
DAY_ID = re.compile(r"rptVotiExcelNuovo_lbGiornata_(\d+)$")


class ArchivePage(HTMLParser):
    """I campi ASP.NET e i pulsanti Excel attivi della pagina."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden: dict[str, str] = {}
        self.days: dict[int, tuple[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "input" and values.get("type") == "hidden" and values.get("name"):
            self.hidden[str(values["name"])] = str(values.get("value") or "")
            return
        if tag != "a":
            return
        match = DAY_ID.search(str(values.get("id") or ""))
        postback = POSTBACK.search(html.unescape(str(values.get("href") or "")))
        if not match or not postback:
            return
        # L'indice HTML parte da zero; la giornata per chi usa il file da uno.
        self.days[int(match.group(1)) + 1] = postback.groups()


def _opener():
    return build_opener(HTTPCookieProcessor(CookieJar()))


def available(opener, url: str) -> ArchivePage:
    response = opener.open(Request(url, headers={"User-Agent": UA}), timeout=30)
    page = ArchivePage()
    page.feed(response.read().decode("utf-8", "replace"))
    if not page.hidden:
        raise ValueError("la pagina voti non contiene i campi ASP.NET attesi")
    return page


def download(opener, url: str, page: ArchivePage, gameweek: int) -> bytes:
    try:
        target, argument = page.days[gameweek]
    except KeyError as error:
        raise ValueError(f"giornata {gameweek} non disponibile") from error
    form = {**page.hidden, "__EVENTTARGET": target, "__EVENTARGUMENT": argument}
    request = Request(
        url,
        data=urlencode(form).encode(),
        headers={
            "User-Agent": UA,
            "Referer": url,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    response = opener.open(request, timeout=60)
    payload = response.read()
    if not payload.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04")):
        raise ValueError("il postback non ha restituito un file Excel")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-2027", help="es. 2026-2027")
    parser.add_argument("--out", type=Path, default=Path("data"), help="radice archivio dati")
    parser.add_argument("--gameweek", type=int, action="append", help="scarica solo questa giornata")
    args = parser.parse_args()

    url = URL.format(season=args.season)
    opener = _opener()
    page = available(opener, url)
    requested = args.gameweek or sorted(page.days)
    if not requested:
        print("nessun voto Excel disponibile", file=sys.stderr)
        return

    # Nell'archivio del progetto le stagioni hanno il formato breve
    # ``2026-27``; Fanta.Soccer usa invece ``2026-2027`` nell'URL.
    start, separator, end = args.season.partition("-")
    if separator != "-" or not (start.isdigit() and len(start) == 4 and end.isdigit() and len(end) == 4):
        raise SystemExit("--season deve avere il formato 2026-2027")
    destination = args.out / f"{start}-{end[-2:]}"
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for gameweek in requested:
        if gameweek not in page.days:
            if args.gameweek:
                raise SystemExit(f"giornata {gameweek} non disponibile")
            continue
        path = destination / f"Voti_{gameweek}a_SerieA.xls"
        if path.exists():
            continue
        path.write_bytes(download(opener, url, page, gameweek))
        downloaded += 1
        print(f"OK: giornata {gameweek} -> {path}")
    if not downloaded:
        print("nessun voto nuovo")


if __name__ == "__main__":
    main()
