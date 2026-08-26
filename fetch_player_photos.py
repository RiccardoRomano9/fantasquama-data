#!/usr/bin/env python3
"""Resolve player identities with Wikimedia and attach verified Gazzetta photos.

Only URLs whose response is a real, non-placeholder PNG are written. Network
errors are isolated: one player or one unavailable source never aborts the run.
Invalid input files and malformed overrides do abort, because publishing a
silently misconfigured mapping would be worse than doing nothing.
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import os
import random
import re
import struct
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

WIKIPEDIA_API = "https://it.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
GAZZETTA_ROOT = "https://images2.gazzettaobjects.it/assets-mc/calcio/giocatori"
CACHE_VERSION = 1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_IMAGE_BYTES = 2_000_000
FOOTBALLER_QIDS = {"Q937857"}
FOOTBALL_WORDS = ("calciator", "footballer", "football player", "futbolista", "futebolista")
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}
TRANSLITERATION = str.maketrans({
    "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "ł": "l", "Ł": "L", "ı": "i", "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE", "ß": "ss", "þ": "th", "Þ": "Th",
})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_time(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def chunks(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start:start + size]


def ascii_text(value: object) -> str:
    text = str(value or "").translate(TRANSLITERATION)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalized_name(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text(value).lower()))


def name_tokens(value: object) -> tuple[str, ...]:
    return tuple(normalized_name(value).split())


def gazzetta_slug(value: object) -> str:
    # Gazzetta normally joins apostrophes (D'Ambrosio -> dambrosio), while
    # spaces and hyphens delimit words. Verification remains authoritative.
    text = ascii_text(value).lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def gazzetta_slug_with_hyphens(value: object) -> str:
    """CDN spelling used for compound surnames such as Loftus-Cheek."""
    text = ascii_text(value).lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9-]+", "_", text)
    return re.sub(r"-+", "-", text).strip("_-")


def photo_url(name: str, birth_date: str) -> str:
    year, month, day = birth_date.split("-")
    return f"{GAZZETTA_ROOT}/{gazzetta_slug(name)}_{day}{month}{year}.png"


def photo_urls(name: str, birth_date: str) -> list[str]:
    """Try the CDN's preserved-hyphen spelling before the legacy normalized one."""
    year, month, day = birth_date.split("-")
    slugs = [gazzetta_slug_with_hyphens(name), gazzetta_slug(name)]
    return [f"{GAZZETTA_ROOT}/{slug}_{day}{month}{year}.png"
            for slug in dict.fromkeys(slugs)]


def provider_id(url: str) -> str:
    return "gazzetta:" + Path(urlparse(url).path).stem


def fingerprint(player: dict[str, Any], override: dict[str, Any] | None) -> str:
    relevant = {
        "id": str(player.get("id", "")), "name": player.get("name"),
        "fullName": player.get("fullName"), "team": player.get("team"),
        "role": player.get("role"), "override": override or {},
    }
    # Invalidate only affected cache entries when the hyphen-preserving URL
    # strategy changes; the other hundreds of verified portraits stay hot.
    if any("-" in str(value or "") for value in (
        player.get("name"), player.get("fullName"),
        (override or {}).get("fullName"), (override or {}).get("gazzettaName"),
    )):
        relevant["hyphenSlugVersion"] = 2
    raw = json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


class FetchError(Exception):
    def __init__(self, message: str, *, status: int | None = None, transient: bool = False):
        super().__init__(message)
        self.status = status
        self.transient = transient


class CircuitOpen(FetchError):
    pass


@dataclass
class Response:
    body: bytes
    headers: Any
    status: int
    url: str


class HTTPClient:
    """Polite, retrying HTTP client with a per-host circuit breaker."""

    def __init__(self, user_agent: str, *, retries: int = 3,
                 wikimedia_delay: float = 0.25, image_delay: float = 0.35):
        self.user_agent = user_agent
        self.retries = retries
        self.delays = {"it.wikipedia.org": wikimedia_delay,
                       "www.wikidata.org": wikimedia_delay,
                       "images2.gazzettaobjects.it": image_delay}
        self.last_request: dict[str, float] = {}
        self.failures: defaultdict[str, int] = defaultdict(int)
        self.api_failures: defaultdict[str, int] = defaultdict(int)
        self.open_hosts: set[str] = set()

    def _pace(self, host: str) -> None:
        wait = self.delays.get(host, 0.25) - (time.monotonic() - self.last_request.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _retry_after(headers: Any) -> float | None:
        value = headers.get("Retry-After") if headers else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(value)
                return max(0.0, (when - datetime.now(when.tzinfo)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    def get(self, url: str, *, params: dict[str, Any] | None = None,
            accept: str = "application/json", limit: int = 5_000_000) -> Response:
        if params:
            url += ("&" if "?" in url else "?") + urlencode(params)
        host = urlparse(url).hostname or ""
        if host in self.open_hosts:
            raise CircuitOpen(f"circuito aperto per {host}", transient=True)
        last_error: FetchError | None = None
        for attempt in range(self.retries + 1):
            self._pace(host)
            request = Request(url, headers={"User-Agent": self.user_agent, "Accept": accept,
                                            "Accept-Encoding": "identity"})
            try:
                with urlopen(request, timeout=25) as response:
                    self.last_request[host] = time.monotonic()
                    body = response.read(limit + 1)
                    if len(body) > limit:
                        raise FetchError(f"risposta oltre {limit} byte", transient=False)
                    self.failures[host] = 0
                    return Response(body, response.headers, response.status, response.geturl())
            except HTTPError as exc:
                self.last_request[host] = time.monotonic()
                transient = exc.code in TRANSIENT_HTTP
                last_error = FetchError(f"HTTP {exc.code} per {url}", status=exc.code,
                                        transient=transient)
                retry_after = self._retry_after(exc.headers)
            except (URLError, TimeoutError, OSError) as exc:
                self.last_request[host] = time.monotonic()
                last_error = FetchError(f"rete: {exc}", transient=True)
                retry_after = None
            if not last_error.transient or attempt == self.retries:
                break
            delay = retry_after if retry_after is not None else 2 ** attempt + random.random() * 0.25
            time.sleep(min(delay, 60.0))
        assert last_error is not None
        if last_error.transient:
            self.failures[host] += 1
            if self.failures[host] >= 3:
                self.open_hosts.add(host)
        raise last_error

    def json(self, url: str, **params: Any) -> dict[str, Any]:
        # 15 still yields to a meaningfully lagged cluster while avoiding a
        # permanent retry loop when Wikidata's normal replica lag is ~10 s.
        query = {"format": "json", "formatversion": 2, "maxlag": 15, **params}
        host = urlparse(url).hostname or ""
        for attempt in range(self.retries + 1):
            response = self.get(url, params=query)
            try:
                value = json.loads(response.body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._record_api_failure(host)
                raise FetchError(f"JSON non valido da {url}: {exc}", transient=True) from exc
            if not isinstance(value, dict):
                self._record_api_failure(host)
                raise FetchError(f"risposta JSON inattesa da {url}", transient=True)
            error = value.get("error")
            if not error:
                self.api_failures[host] = 0
                return value
            code = str(error.get("code", "error"))
            if code not in {"maxlag", "ratelimited", "readonly"} or attempt == self.retries:
                self._record_api_failure(host)
                raise FetchError(f"API {code}: {error.get('info', '')}", transient=True)
            retry_after = self._retry_after(response.headers)
            lag = error.get("lag") if isinstance(error.get("lag"), (int, float)) else None
            time.sleep(min(retry_after or lag or 2 ** attempt, 60.0))
        raise AssertionError("ciclo retry API terminato senza risultato")

    def _record_api_failure(self, host: str) -> None:
        self.api_failures[host] += 1
        if self.api_failures[host] >= 3:
            self.open_hosts.add(host)


def title_key(value: object) -> str:
    return " ".join(str(value or "").replace("_", " ").split()).casefold()


class Wikimedia:
    def __init__(self, client: HTTPClient):
        self.client = client
        self.had_transient = False

    def page_qids(self, titles: list[str]) -> dict[str, str]:
        """Map requested titles to QIDs, following normalization and redirects."""
        output: dict[str, str] = {}
        for part in chunks(sorted(set(filter(None, titles))), 20):
            try:
                data = self.client.json(WIKIPEDIA_API, action="query", prop="pageprops",
                                        ppprop="wikibase_item", redirects=1, titles="|".join(part))
            except FetchError:
                self.had_transient = True
                continue
            query = data.get("query", {})
            links: dict[str, str] = {}
            for kind in ("normalized", "redirects"):
                for row in query.get(kind, []) or []:
                    links[title_key(row.get("from"))] = title_key(row.get("to"))
            by_title = {
                title_key(page.get("title")): page.get("pageprops", {}).get("wikibase_item")
                for page in query.get("pages", []) or [] if not page.get("missing")
            }
            for original in part:
                key, seen = title_key(original), set()
                while key in links and key not in seen:
                    seen.add(key)
                    key = links[key]
                if by_title.get(key):
                    output[original] = str(by_title[key])
        return output

    def search_titles(self, name: str) -> list[str]:
        try:
            data = self.client.json(WIKIPEDIA_API, action="query", list="search", srnamespace=0,
                                    srlimit=5, srsearch=f'"{name}" calciatore')
        except FetchError:
            self.had_transient = True
            return []
        return [str(row["title"]) for row in data.get("query", {}).get("search", []) if row.get("title")]

    def search_entities(self, name: str) -> list[str]:
        """Fallback for players who have a Wikidata item but no itwiki page."""
        try:
            data = self.client.json(WIKIDATA_API, action="wbsearchentities", search=name,
                                    language="it", uselang="it", type="item", limit=5)
        except FetchError:
            self.had_transient = True
            return []
        return [str(row["id"]) for row in data.get("search", []) if row.get("id")]

    def entities(self, qids: Iterable[str]) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for part in chunks(sorted(set(filter(None, qids))), 40):
            try:
                data = self.client.json(WIKIDATA_API, action="wbgetentities", ids="|".join(part),
                                        props="labels|aliases|descriptions|claims",
                                        languages="it|en", languagefallback=1)
            except FetchError:
                self.had_transient = True
                continue
            output.update({str(qid): entity for qid, entity in data.get("entities", {}).items()
                           if not entity.get("missing")})
        return output


def claim_ids(entity: dict[str, Any], prop: str) -> set[str]:
    values: set[str] = set()
    for claim in entity.get("claims", {}).get(prop, []) or []:
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and value.get("id"):
            values.add(str(value["id"]))
    return values


def birth_date(entity: dict[str, Any]) -> str | None:
    claims = entity.get("claims", {}).get("P569", []) or []
    claims = sorted(claims, key=lambda row: row.get("rank") == "preferred", reverse=True)
    for claim in claims:
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        raw, precision = value.get("time", ""), value.get("precision", 0)
        if precision >= 11 and re.match(r"^\+\d{4}-\d{2}-\d{2}T", raw):
            return raw[1:11]
    return None


def language_value(entity: dict[str, Any], field: str, language: str) -> str | None:
    value = entity.get(field, {}).get(language)
    return str(value.get("value")) if isinstance(value, dict) and value.get("value") else None


def entity_names(entity: dict[str, Any], title: str | None = None) -> list[str]:
    names = [language_value(entity, "labels", "it"), language_value(entity, "labels", "en"), title]
    for language in ("it", "en"):
        names.extend(row.get("value") for row in entity.get("aliases", {}).get(language, []) or [])
    return list(dict.fromkeys(str(value) for value in names if value))


def is_footballer(entity: dict[str, Any]) -> bool:
    if "Q5" not in claim_ids(entity, "P31"):
        return False
    if claim_ids(entity, "P106") & FOOTBALLER_QIDS:
        return True
    descriptions = " ".join(filter(None, [language_value(entity, "descriptions", "it"),
                                             language_value(entity, "descriptions", "en")])).lower()
    return any(word in descriptions for word in FOOTBALL_WORDS)


def name_score(player: dict[str, Any], candidate_names: list[str]) -> int:
    full, short = name_tokens(player.get("fullName")), name_tokens(player.get("name"))
    best = 0
    for candidate in map(name_tokens, candidate_names):
        if not candidate:
            continue
        if full:
            if candidate == full:
                best = max(best, 100)
            elif len(candidate) >= 2 and set(candidate).issubset(full):
                best = max(best, 88)
            elif len(full) >= 2 and set(full).issubset(candidate):
                best = max(best, 84)
        if short and candidate == short and len(short) >= 2:
            best = max(best, 82)
        elif short and len(short) >= 2 and set(short).issubset(candidate):
            best = max(best, 78)
        initial = re.search(r"\b([A-Za-z]{1,4})\.\s*$", str(player.get("name", "")))
        if initial and short and short[0] in candidate and any(
            token.startswith(initial.group(1).lower()) for token in candidate
        ):
            best = max(best, 72)
    return best


def choose_entity(player: dict[str, Any], candidates: list[tuple[str, str | None]],
                  entities: dict[str, dict[str, Any]], forced_qid: str | None = None) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for qid, title in dict.fromkeys(candidates):
        entity = entities.get(qid)
        if not entity or not is_footballer(entity) or not birth_date(entity):
            continue
        names = entity_names(entity, title)
        score = 1000 if forced_qid == qid else name_score(player, names)
        ranked.append({"qid": qid, "title": title, "entity": entity, "names": names, "score": score})
    ranked.sort(key=lambda row: (-row["score"], row["qid"]))
    if not ranked:
        return {"status": "not_found", "candidates": []}
    if ranked[0]["score"] < 80:
        return {"status": "ambiguous", "candidates": [row["qid"] for row in ranked[:5]]}
    if len(ranked) > 1 and ranked[1]["score"] >= ranked[0]["score"] - 5:
        return {"status": "ambiguous", "candidates": [row["qid"] for row in ranked[:5]]}
    winner = ranked[0]
    canonical = (language_value(winner["entity"], "labels", "it")
                 or language_value(winner["entity"], "labels", "en") or winner["names"][0])
    return {"status": "resolved", "wikidataID": winner["qid"],
            "wikipediaTitle": winner["title"], "fullName": canonical,
            "birthDate": birth_date(winner["entity"])}


def parse_png(content_type: str, body: bytes, invalid_hashes: set[str]) -> dict[str, Any]:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime not in {"image/png", "image/x-png"}:
        raise ValueError(f"Content-Type {mime or 'mancante'}")
    if len(body) < 1024 or not body.startswith(PNG_SIGNATURE):
        raise ValueError("contenuto non PNG o troppo piccolo")
    if body[12:16] != b"IHDR" or len(body) < 33:
        raise ValueError("IHDR PNG mancante")
    width, height = struct.unpack(">II", body[16:24])
    if not (80 <= width <= 4000 and 80 <= height <= 4000):
        raise ValueError(f"dimensioni sospette {width}x{height}")
    if b"IEND" not in body[-32:]:
        raise ValueError("PNG troncato")
    digest = hashlib.sha256(body).hexdigest()
    if digest in invalid_hashes:
        raise ValueError(f"placeholder noto sha256:{digest}")
    return {"sha256": digest, "width": width, "height": height,
            "bytes": len(body), "contentType": mime}


def validate_photo(client: HTTPClient, url: str, invalid_hashes: set[str]) -> tuple[str, Any]:
    try:
        response = client.get(url, accept="image/png", limit=MAX_IMAGE_BYTES)
    except FetchError as exc:
        if exc.status == 404:
            return "404", "HTTP 404"
        return ("transient" if exc.transient else "invalid"), str(exc)
    try:
        return "valid", parse_png(response.headers.get("Content-Type", ""), response.body, invalid_hashes)
    except ValueError as exc:
        return "invalid", str(exc)


def candidate_names(player: dict[str, Any], resolved: dict[str, Any], override: dict[str, Any]) -> list[str]:
    values = [override.get("gazzettaName"), resolved.get("fullName"),
              resolved.get("wikipediaTitle"), player.get("fullName")]
    output: list[str] = []
    for raw in values:
        if not raw:
            continue
        value = re.sub(r"\s*\([^)]*\)\s*$", "", str(raw)).strip()
        if value and value not in output:
            output.append(value)
        tokens = value.split()
        if len(tokens) > 2:
            shorter = f"{tokens[0]} {tokens[-1]}"
            if shorter not in output:
                output.append(shorter)
    return output


def cache_entry(player: dict[str, Any], override: dict[str, Any], status: str,
                now: datetime, **extra: Any) -> dict[str, Any]:
    retry_days = 30 if status in {"not_found", "ambiguous", "gazzetta_404", "image_invalid"} else None
    entry: dict[str, Any] = {"inputHash": fingerprint(player, override), "status": status,
                             "checkedAt": iso(now)}
    if retry_days:
        entry["retryAfter"] = iso(now + timedelta(days=retry_days))
    entry.update({key: value for key, value in extra.items() if value is not None})
    return entry


def validate_overrides(data: Any, player_ids: set[str]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    if not isinstance(data, dict) or data.get("version", 1) != 1:
        raise ValueError("gli override devono essere un oggetto con version: 1")
    players = data.get("players", {})
    if not isinstance(players, dict):
        raise ValueError("overrides.players deve essere un oggetto indicizzato per id")
    allowed = {"skip", "wikidataID", "fullName", "birthDate", "gazzettaName",
               "photoURL", "photoProviderID", "note"}
    for pid, row in players.items():
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError(f"override {pid}: campi non validi {sorted(set(row or {}) - allowed)}")
        qid = row.get("wikidataID")
        if qid and not re.fullmatch(r"Q\d+", str(qid)):
            raise ValueError(f"override {pid}: wikidataID non valido")
        birth = row.get("birthDate")
        if birth:
            try:
                datetime.strptime(str(birth), "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(f"override {pid}: birthDate non valida") from exc
        if not any((row.get("skip"), qid, row.get("photoURL"),
                    birth and (row.get("gazzettaName") or row.get("fullName")))):
            raise ValueError(f"override {pid}: indicare skip, wikidataID, photoURL o nome+data")
    hashes = data.get("invalidImageSHA256", [])
    if not isinstance(hashes, list) or any(not re.fullmatch(r"[0-9a-fA-F]{64}", str(v)) for v in hashes):
        raise ValueError("invalidImageSHA256 deve contenere hash SHA-256 esadecimali")
    # Old roster IDs may remain deliberately: keeping historical decisions is
    # useful across transfers and seasons. They are validated, then ignored.
    current = {str(key): value for key, value in players.items() if str(key) in player_ids}
    return current, {str(v).lower() for v in hashes}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists() and default is not None:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: JSON non valido: {exc}") from exc


def write_json_if_changed(path: Path, value: Any, *, indent: int) -> bool:
    rendered = json.dumps(value, ensure_ascii=False, indent=indent) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return True


def fresh(entry: dict[str, Any] | None, wanted_hash: str, now: datetime,
          refresh_valid_days: int, refresh_all: bool) -> bool:
    if refresh_all or not entry or entry.get("inputHash") != wanted_hash:
        return False
    if entry.get("status") == "valid":
        checked = parse_time(entry.get("checkedAt"))
        return bool(checked and checked + timedelta(days=refresh_valid_days) > now)
    if entry.get("status") == "manual_fallback":
        return True
    retry_after = parse_time(entry.get("retryAfter"))
    return bool(retry_after and retry_after > now)


def resolve_players(players: list[dict[str, Any]], due: list[dict[str, Any]],
                    overrides: dict[str, dict[str, Any]], cache: dict[str, Any],
                    wikimedia: Wikimedia, client: HTTPClient, invalid_hashes: set[str],
                    now: datetime, transient: list[str]) -> None:
    """Resolve and validate due players, mutating only cache['players']."""
    cached = cache["players"]
    by_full: defaultdict[str, list[str]] = defaultdict(list)
    for player in players:
        if player.get("fullName"):
            by_full[normalized_name(player["fullName"])].append(str(player["id"]))

    contexts: dict[str, dict[str, Any]] = {}
    exact_titles: list[str] = []
    for player in due:
        pid, override = str(player["id"]), overrides.get(str(player["id"]), {})
        if override.get("skip"):
            cached[pid] = cache_entry(player, override, "manual_fallback", now,
                                      note=override.get("note"))
            continue
        duplicate = (player.get("fullName")
                     and len(by_full[normalized_name(player["fullName"])]) > 1)
        if duplicate and not override:
            cached[pid] = cache_entry(
                player, override, "ambiguous", now,
                reason="fullName duplicato nel listone",
                candidates=by_full[normalized_name(player["fullName"])],
            )
            continue
        direct = override.get("photoURL")
        if direct:
            contexts[pid] = {
                "player": player, "override": override,
                "resolved": {"status": "resolved", "fullName": override.get("fullName")},
                "urls": [str(direct)],
            }
            continue
        if override.get("birthDate") and (override.get("gazzettaName") or override.get("fullName")):
            contexts[pid] = {"player": player, "override": override, "resolved": {
                "status": "resolved", "fullName": override.get("fullName") or override.get("gazzettaName"),
                "birthDate": override["birthDate"],
            }}
            continue
        previous = cached.get(pid, {})
        reusable_identity = (
            previous.get("inputHash") == fingerprint(player, override)
            and previous.get("status") in {"valid", "gazzetta_404", "image_invalid"}
            and previous.get("fullName") and previous.get("birthDate")
        )
        if reusable_identity:
            resolved = {field: previous.get(field) for field in (
                "fullName", "birthDate", "wikidataID", "wikipediaTitle"
            )}
            resolved["status"] = "resolved"
            contexts[pid] = {"player": player, "override": override, "resolved": resolved}
            # A positive cache refresh checks the known URL directly. A
            # negative refresh retries all name variants without re-querying
            # Wikimedia: identity and image availability have different TTLs.
            if previous.get("status") == "valid" and previous.get("photoURL"):
                contexts[pid]["urls"] = [previous["photoURL"]]
            continue
        query = str(player.get("fullName") or player.get("name") or "").strip()
        if not query:
            cached[pid] = cache_entry(player, override, "not_found", now, reason="nome vuoto")
            continue
        contexts[pid] = {"player": player, "override": override, "query": query,
                         "candidates": [], "forced": override.get("wikidataID")}
        if override.get("wikidataID"):
            contexts[pid]["candidates"].append((str(override["wikidataID"]), None))
        else:
            exact_titles.append(query)

    exact = wikimedia.page_qids(exact_titles) if exact_titles else {}
    for context in contexts.values():
        query = context.get("query")
        if query and exact.get(query):
            context["candidates"].append((exact[query], query))

    initial_qids = [qid for context in contexts.values()
                    for qid, _ in context.get("candidates", [])]
    entities = wikimedia.entities(initial_qids) if initial_qids else {}
    unresolved: list[dict[str, Any]] = []
    for context in contexts.values():
        if "resolved" in context:
            continue
        result = choose_entity(context["player"], context["candidates"], entities,
                               context.get("forced"))
        if result["status"] == "resolved" or context.get("forced"):
            context["resolved"] = result
        else:
            unresolved.append(context)

    # Search is the expensive fallback and runs only for exact-title misses.
    search_titles: list[str] = []
    for context in unresolved:
        titles = wikimedia.search_titles(context["query"])
        context["searchTitles"] = titles
        search_titles.extend(titles)
        if not titles:
            context["candidates"].extend(
                (qid, None) for qid in wikimedia.search_entities(context["query"])
            )
    searched_qids = wikimedia.page_qids(search_titles) if search_titles else {}
    for context in unresolved:
        context["candidates"].extend(
            (searched_qids[title], title) for title in context.get("searchTitles", [])
            if title in searched_qids
        )
    fallback_qids = [qid for context in unresolved
                     for qid, _ in context.get("candidates", [])]
    if fallback_qids:
        entities.update(wikimedia.entities(fallback_qids))
    for context in unresolved:
        context["resolved"] = choose_entity(
            context["player"], context["candidates"], entities, context.get("forced")
        )

    for pid, context in contexts.items():
        player, override, resolved = context["player"], context["override"], context["resolved"]
        if resolved.get("status") != "resolved":
            # Do not fossilize a source outage as a genuine negative result.
            if (wikimedia.had_transient or "it.wikipedia.org" in client.open_hosts
                    or "www.wikidata.org" in client.open_hosts):
                transient.append(f"{player.get('name')} — Wikimedia non disponibile")
                continue
            cached[pid] = cache_entry(player, override, resolved["status"], now,
                                      candidates=resolved.get("candidates"))
            continue

        urls = context.get("urls") or [
            url for name in candidate_names(player, resolved, override)
            for url in photo_urls(name, resolved["birthDate"])
        ]
        urls = list(dict.fromkeys(urls))
        attempts: list[dict[str, Any]] = []
        winner: tuple[str, dict[str, Any]] | None = None
        had_transient = False
        for url in urls:
            status, detail = validate_photo(client, url, invalid_hashes)
            attempts.append({"url": url, "status": status,
                             "detail": detail if isinstance(detail, str) else None})
            if status == "valid":
                winner = (url, detail)
                break
            if status == "transient":
                had_transient = True
                break
        common = {
            "fullName": resolved.get("fullName"), "birthDate": resolved.get("birthDate"),
            "wikidataID": resolved.get("wikidataID"),
            "wikipediaTitle": resolved.get("wikipediaTitle"), "attempts": attempts,
            "note": override.get("note"),
        }
        if winner:
            url, image = winner
            cached[pid] = cache_entry(
                player, override, "valid", now, **common, photoURL=url,
                photoProviderID=override.get("photoProviderID") or provider_id(url), image=image,
            )
        elif had_transient:
            transient.append(f"{player.get('name')} — {attempts[-1]['detail']}")
            # Preserve an older valid entry and the JSON fields on transient errors.
        else:
            status = ("gazzetta_404" if attempts and all(a["status"] == "404" for a in attempts)
                      else "image_invalid")
            cached[pid] = cache_entry(player, override, status, now, **common)


def reject_shared_images(players: list[dict[str, Any]], cache: dict[str, Any],
                         overrides: dict[str, dict[str, Any]], now: datetime) -> None:
    """An identical portrait served at different URLs is a provider placeholder."""
    by_hash: defaultdict[str, list[str]] = defaultdict(list)
    player_by_id = {str(player["id"]): player for player in players}
    for pid, entry in cache["players"].items():
        current = player_by_id.get(pid)
        if (current and entry.get("inputHash") == fingerprint(current, overrides.get(pid))
                and entry.get("status") == "valid" and entry.get("image", {}).get("sha256")):
            by_hash[entry["image"]["sha256"]].append(pid)
    for digest, pids in by_hash.items():
        distinct_urls = {cache["players"][pid].get("photoURL") for pid in pids}
        if len(pids) < 2 or len(distinct_urls) < 2:
            continue
        for pid in pids:
            old, player = cache["players"][pid], player_by_id.get(pid)
            if not player:
                continue
            cache["players"][pid] = cache_entry(
                player, overrides.get(pid, {}), "image_invalid", now,
                fullName=old.get("fullName"), birthDate=old.get("birthDate"),
                wikidataID=old.get("wikidataID"), wikipediaTitle=old.get("wikipediaTitle"),
                attempts=[{"url": old.get("photoURL"), "status": "invalid",
                           "detail": f"immagine condivisa/placeholder sha256:{digest}"}],
            )


def apply_cache(players: list[dict[str, Any]], cache: dict[str, Any],
                overrides: dict[str, dict[str, Any]]) -> tuple[int, int]:
    valid = fallback = 0
    for player in players:
        pid = str(player["id"])
        entry = cache["players"].get(pid)
        matches = bool(entry and entry.get("inputHash") == fingerprint(player, overrides.get(pid)))
        if matches and entry.get("fullName") and not player.get("fullName"):
            player["fullName"] = entry["fullName"]
            # Adding the discovered name changes the input by design. Advance
            # the fingerprint with it so the next run is still a cache hit.
            entry["inputHash"] = fingerprint(player, overrides.get(pid))
        if matches and entry.get("status") == "valid":
            player["photoURL"] = entry["photoURL"]
            player["photoProviderID"] = entry["photoProviderID"]
            valid += 1
        else:
            # No matching cache entry can mean a transient outage: retain a
            # previously published URL, but do not count it as verified.
            if matches:
                player.pop("photoURL", None)
                player.pop("photoProviderID", None)
            fallback += 1
    return valid, fallback


def sync_derived(base_players: list[dict[str, Any]], derived: dict[str, Any],
                 updated_at: str | None = None) -> bool:
    """Copy only identity/photo fields, preserving the derived live-lineup data."""
    if not isinstance(derived, dict) or not isinstance(derived.get("players"), list):
        raise ValueError("il file derivato non contiene l'array players")
    source = {str(player["id"]): player for player in base_players}
    target_ids = {str(player.get("id", "")) for player in derived["players"]}
    if target_ids != set(source):
        missing = sorted(set(source) - target_ids)
        extra = sorted(target_ids - set(source))
        raise ValueError(f"base e derivato hanno rose diverse (mancanti={missing}, extra={extra})")
    changed = False
    for player in derived["players"]:
        original = source[str(player["id"])]
        for field in ("fullName", "photoURL", "photoProviderID"):
            if original.get(field) is not None:
                changed |= player.get(field) != original[field]
                player[field] = original[field]
            else:
                changed |= field in player
                player.pop(field, None)
    if changed and updated_at is not None:
        derived["generatedAt"] = updated_at
    return changed


def report(players: list[dict[str, Any]], cache: dict[str, Any],
           overrides: dict[str, dict[str, Any]], valid: int, fallback: int,
           transient: list[str], cache_hits: int, lookups: int) -> str:
    player_by_id = {str(player["id"]): player for player in players}
    names = {pid: str(player.get("fullName") or player.get("name") or pid)
             for pid, player in player_by_id.items()}
    current = {
        pid: entry for pid, entry in cache["players"].items()
        if pid in names and entry.get("inputHash") == fingerprint(player_by_id[pid], overrides.get(pid))
    }
    ambiguous = [names[pid] for pid, entry in current.items() if entry.get("status") == "ambiguous"]
    missing = [names[pid] for pid, entry in current.items()
               if entry.get("status") in {"not_found", "manual_fallback"}]
    invalid = [names[pid] for pid, entry in current.items() if entry.get("status") == "image_invalid"]
    not_found_urls = sorted({attempt["url"] for entry in current.values()
                             for attempt in entry.get("attempts", [])
                             if attempt.get("status") == "404"})
    lines = ["## Foto giocatori", "", f"- URL PNG validi: **{valid}**",
             f"- Fallback locale: **{fallback}**",
             f"- Cache riusata: **{cache_hits}**; giocatori elaborati: **{lookups}**"]
    for heading, values in (
        ("Ambigui", ambiguous), ("Non trovati / override fallback", missing),
        ("Immagini non valide o placeholder", invalid),
        ("Errori transitori (non memorizzati come esiti negativi)", transient),
        ("URL Gazzetta con HTTP 404", not_found_urls),
    ):
        lines.extend(["", f"### {heading}", ""])
        lines.extend([f"- {value}" for value in sorted(set(values))] or ["- Nessuno"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("player-photo-cache.json"))
    parser.add_argument("--overrides", type=Path, default=Path("player-photo-overrides.json"))
    parser.add_argument("--derived", type=Path,
                        help="sincronizza i campi foto nel serieA.json già generato")
    parser.add_argument("--report", type=Path, help="scrive anche il report Markdown qui")
    parser.add_argument("--refresh-valid-days", type=int, default=90)
    parser.add_argument("--refresh-all", action="store_true")
    parser.add_argument("--user-agent", default=os.environ.get(
        "PLAYER_PHOTO_USER_AGENT",
        "FantaSquamaPhotoBot/1.0 (https://github.com/RiccardoRomano9/FantaSquama; contact via GitHub)",
    ))
    parser.add_argument("--wikimedia-delay", type=float, default=0.25)
    parser.add_argument("--image-delay", type=float, default=0.35)
    args = parser.parse_args(argv)
    if args.refresh_valid_days < 1 or args.wikimedia_delay < 0 or args.image_delay < 0:
        parser.error("giorni e intervalli devono essere positivi")

    base = read_json(args.base)
    if not isinstance(base, dict) or not isinstance(base.get("players"), list):
        raise ValueError(f"{args.base}: manca l'array players")
    players = base["players"]
    ids = [str(player.get("id", "")) for player in players]
    if any(not pid for pid in ids) or len(ids) != len(set(ids)):
        raise ValueError("gli id giocatore devono essere presenti e univoci")

    override_data = read_json(args.overrides,
                              {"version": 1, "players": {}, "invalidImageSHA256": []})
    overrides, invalid_hashes = validate_overrides(override_data, set(ids))
    cache = read_json(args.cache, {"version": CACHE_VERSION, "players": {}})
    if (not isinstance(cache, dict) or cache.get("version") != CACHE_VERSION
            or not isinstance(cache.get("players"), dict)):
        raise ValueError(f"{args.cache}: formato cache non supportato (atteso version {CACHE_VERSION})")

    now = utcnow()
    due, cache_hits = [], 0
    for player in players:
        pid = str(player["id"])
        wanted = fingerprint(player, overrides.get(pid))
        if fresh(cache["players"].get(pid), wanted, now,
                 args.refresh_valid_days, args.refresh_all):
            cache_hits += 1
        else:
            due.append(player)

    client = HTTPClient(args.user_agent, wikimedia_delay=args.wikimedia_delay,
                        image_delay=args.image_delay)
    transient: list[str] = []
    resolve_players(players, due, overrides, cache, Wikimedia(client), client,
                    invalid_hashes, now, transient)
    reject_shared_images(players, cache, overrides, now)
    valid, fallback = apply_cache(players, cache, overrides)

    base_changed = write_json_if_changed(args.base, base, indent=1)
    derived_changed = False
    if args.derived:
        derived = read_json(args.derived)
        sync_derived(players, derived, iso(now))
        derived_changed = write_json_if_changed(args.derived, derived, indent=1)
    cache_changed = write_json_if_changed(args.cache, cache, indent=2)
    summary = report(players, cache, overrides, valid, fallback, transient, cache_hits, len(due))
    print(summary, end="")
    if args.report:
        args.report.write_text(summary, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as handle:
            handle.write(summary)
    print(f"File modificati: base={'sì' if base_changed else 'no'}, "
          f"derivato={'sì' if derived_changed else 'no'}, "
          f"cache={'sì' if cache_changed else 'no'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"errore di configurazione: {exc}", file=sys.stderr)
        raise SystemExit(2)
