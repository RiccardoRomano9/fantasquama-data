"""Le notizie di calcio, da piu' feed RSS, in un solo file.

    python fetch_news.py -o data/news.json

Sta fuori dal pacchetto come gli altri `fetch_*`: `fantasquama/` non fa
chiamate di rete. Solo stdlib, nessuna dipendenza.

Ogni fonte e' una riga in FEEDS. **Un feed che non risponde o che oggi e'
vuoto non e' un errore**: i giornali cambiano indirizzo, svuotano il feed
per manutenzione, o sbagliano un deploy. Si prende quello che c'e' e si
scrive quante fonti hanno risposto; ci si ferma solo se non risponde
nessuno, perche' li' il problema e' la rete e non il giornale.

Dell'articolo si tiene solo cio' che serve a decidere se aprirlo: titolo,
sommario, firma, data, immagine e **il collegamento**. Il testo no -- e'
roba loro, e l'articolo si legge da loro.
"""

import argparse
import html
import json
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

FEEDS: tuple[tuple[str, str], ...] = (
    ("SOS Fanta", "https://www.sosfanta.com/feed"),
    ("Gazzetta dello Sport", "https://www.gazzetta.it/rss/calcio.xml"),
    # Provati e oggi vuoti o irraggiungibili. Restano scritti perche' domani
    # potrebbero tornare, e perche' l'elenco dice anche cosa si e' cercato:
    # ("Tuttosport", "https://www.tuttosport.com/rss/calcio.xml"),
)

UA = "FantaSquama/1.0 (lettore RSS a uso personale)"

# Piu' vecchio di cosi' non e' una notizia. Serve soprattutto contro i feed
# che mescolano l'archivio all'attualita': senza, in cima finisce un pezzo
# del 2024 solo perche' quel giornale ordina gli item a modo suo.
MAX_AGE = timedelta(days=5)
MAX_ITEMS = 40

NS = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "media": "http://search.yahoo.com/mrss/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


def _testo(item: ET.Element, percorso: str) -> str:
    trovato = item.find(percorso, NS)
    return html.unescape((trovato.text or "").strip()) if trovato is not None else ""


def _immagine(item: ET.Element) -> str:
    """L'illustrazione, cercata dove i vari feed la mettono."""
    # `or` su un Element e' deprecato: un elemento senza figli e' falso anche
    # quando esiste, quindi il confronto va fatto con None
    media = item.find("media:content", NS)
    if media is None:
        media = item.find("media:thumbnail", NS)
    if media is not None and media.attrib.get("url"):
        return media.attrib["url"]
    allegato = item.find("enclosure")
    if allegato is not None and str(allegato.attrib.get("type", "")).startswith("image"):
        return allegato.attrib.get("url", "")
    # ultima risorsa: la prima <img> del corpo dell'articolo
    corpo = _testo(item, "content:encoded")
    trovata = re.search(r'<img[^>]+src="([^"]+)"', corpo)
    return trovata.group(1) if trovata else ""


def _quando(item: ET.Element) -> datetime | None:
    grezza = _testo(item, "pubDate") or _testo(item, "dc:date")
    if not grezza:
        return None
    try:
        quando = parsedate_to_datetime(grezza)
    except (TypeError, ValueError):
        return None
    return quando if quando.tzinfo else quando.replace(tzinfo=timezone.utc)


def _pulisci(testo: str) -> str:
    """Via i tag: il sommario di certi feed e' un pezzo di HTML."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", testo)).strip()


def leggi(fonte: str, url: str, adesso: datetime) -> list[dict]:
    richiesta = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(richiesta, timeout=25) as risposta:
        radice = ET.fromstring(risposta.read())

    notizie = []
    for item in radice.iter("item"):
        quando = _quando(item)
        if quando is None or adesso - quando > MAX_AGE:
            continue
        titolo = _testo(item, "title")
        collegamento = _testo(item, "link")
        if not titolo or not collegamento:
            continue
        notizie.append({
            "title": titolo,
            "summary": _pulisci(_testo(item, "description"))[:280],
            "author": _testo(item, "dc:creator"),
            "source": fonte,
            "url": collegamento,
            "image": _immagine(item),
            "date": quando.astimezone(timezone.utc).isoformat(timespec="seconds"),
        })
    return notizie


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", help="salva il JSON in questo file")
    args = parser.parse_args()

    adesso = datetime.now(timezone.utc)
    notizie: list[dict] = []
    risposte = 0
    for fonte, url in FEEDS:
        try:
            trovate = leggi(fonte, url, adesso)
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as errore:
            print(f"  {fonte}: non letto ({errore})", file=sys.stderr)
            continue
        risposte += 1
        notizie += trovate
        print(f"  {fonte}: {len(trovate)} notizie recenti", file=sys.stderr)

    if not risposte:
        raise SystemExit("nessun feed ha risposto: e' la rete, non i giornali")

    # in cima le piu' recenti, e mai due volte lo stesso articolo
    viste: set[str] = set()
    ordinate = []
    for notizia in sorted(notizie, key=lambda n: n["date"], reverse=True):
        if notizia["url"] in viste:
            continue
        viste.add(notizia["url"])
        ordinate.append(notizia)

    payload = ordinate[:MAX_ITEMS]
    testo = json.dumps(payload, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            file.write(testo + "\n")
        print(f"OK: {len(payload)} notizie da {risposte} fonti -> {args.out}", file=sys.stderr)
    else:
        print(testo)


if __name__ == "__main__":
    main()
