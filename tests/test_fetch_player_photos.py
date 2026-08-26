import hashlib
import struct
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fetch_player_photos import (
    PNG_SIGNATURE,
    apply_cache,
    birth_date,
    cache_entry,
    choose_entity,
    fresh,
    gazzetta_slug,
    parse_png,
    photo_url,
    resolve_players,
    sync_derived,
    validate_overrides,
)


def claim(value, *, precision=None):
    payload = {"id": value} if precision is None else {
        "time": value, "precision": precision,
    }
    return {"rank": "normal", "mainsnak": {"datavalue": {"value": payload}}}


def footballer(label, born="+2000-04-05T00:00:00Z"):
    return {
        "labels": {"it": {"value": label}},
        "aliases": {},
        "descriptions": {"it": {"value": "calciatore italiano"}},
        "claims": {
            "P31": [claim("Q5")],
            "P106": [claim("Q937857")],
            "P569": [claim(born, precision=11)],
        },
    }


def fake_png(width=370, height=444):
    ihdr = struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
    ihdr += b"\x08\x06\x00\x00\x00" + b"\x00\x00\x00\x00"
    padding = b"x" * 1000
    iend = struct.pack(">I", 0) + b"IEND" + b"\x00\x00\x00\x00"
    return PNG_SIGNATURE + ihdr + padding + iend


class NamesTests(unittest.TestCase):
    def test_slug_handles_accents_apostrophes_hyphens_and_special_letters(self):
        self.assertEqual(gazzetta_slug("M'Bala N'Zola"), "mbala_nzola")
        self.assertEqual(gazzetta_slug("Rasmus Højlund-García"), "rasmus_hojlund_garcia")

    def test_url_uses_gazzetta_date_order(self):
        self.assertEqual(
            photo_url("Nikola Krstović", "2000-04-05"),
            "https://images2.gazzettaobjects.it/assets-mc/calcio/giocatori/nikola_krstovic_05042000.png",
        )


class WikidataTests(unittest.TestCase):
    def test_birth_requires_day_precision(self):
        self.assertEqual(birth_date(footballer("Mario Rossi")), "2000-04-05")
        entity = footballer("Mario Rossi")
        entity["claims"]["P569"] = [claim("+2000-00-00T00:00:00Z", precision=9)]
        self.assertIsNone(birth_date(entity))

    def test_unique_exact_name_is_selected(self):
        player = {"name": "Rossi M.", "fullName": "Mario Rossi"}
        result = choose_entity(player, [("Q10", "Mario Rossi")],
                               {"Q10": footballer("Mario Rossi")})
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["wikidataID"], "Q10")

    def test_equal_candidates_remain_ambiguous(self):
        player = {"name": "Rossi M.", "fullName": "Mario Rossi"}
        result = choose_entity(
            player, [("Q10", "Mario Rossi"), ("Q20", "Mario Rossi")],
            {"Q10": footballer("Mario Rossi"), "Q20": footballer("Mario Rossi")},
        )
        self.assertEqual(result["status"], "ambiguous")

    def test_non_footballer_is_rejected(self):
        entity = footballer("Mario Rossi")
        entity["claims"]["P106"] = [claim("Q82955")]
        entity["descriptions"] = {"it": {"value": "politico italiano"}}
        result = choose_entity({"fullName": "Mario Rossi"}, [("Q10", None)], {"Q10": entity})
        self.assertEqual(result["status"], "not_found")


class ImageTests(unittest.TestCase):
    def test_real_png_shape_is_accepted(self):
        body = fake_png()
        image = parse_png("image/png; charset=binary", body, set())
        self.assertEqual((image["width"], image["height"]), (370, 444))
        self.assertEqual(image["sha256"], hashlib.sha256(body).hexdigest())

    def test_html_disguised_as_success_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Content-Type"):
            parse_png("text/html", b"<html>not found</html>" * 100, set())

    def test_known_placeholder_hash_is_rejected(self):
        body = fake_png()
        with self.assertRaisesRegex(ValueError, "placeholder"):
            parse_png("image/png", body, {hashlib.sha256(body).hexdigest()})

    def test_tiny_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensioni"):
            parse_png("image/png", fake_png(1, 1), set())


class CacheAndOverridesTests(unittest.TestCase):
    def test_negative_cache_expires_but_manual_fallback_does_not(self):
        now = datetime(2026, 8, 26, tzinfo=timezone.utc)
        player = {"id": "1", "name": "Rossi", "team": "Roma", "role": "D"}
        negative = cache_entry(player, {}, "not_found", now)
        manual = cache_entry(player, {"skip": True}, "manual_fallback", now)
        self.assertTrue(fresh(negative, negative["inputHash"], now, 90, False))
        self.assertFalse(fresh(negative, negative["inputHash"],
                               datetime(2026, 10, 1, tzinfo=timezone.utc), 90, False))
        self.assertTrue(fresh(manual, manual["inputHash"],
                              datetime(2030, 1, 1, tzinfo=timezone.utc), 90, False))

    def test_discovered_name_advances_fingerprint(self):
        now = datetime(2026, 8, 26, tzinfo=timezone.utc)
        player = {"id": "1", "name": "Rossi", "fullName": None,
                  "team": "Roma", "role": "D"}
        entry = cache_entry(player, {}, "valid", now, fullName="Mario Rossi",
                            photoURL="https://example.test/mario.png",
                            photoProviderID="gazzetta:mario")
        cache = {"version": 1, "players": {"1": entry}}
        self.assertEqual(apply_cache([player], cache, {}), (1, 0))
        self.assertEqual(player["fullName"], "Mario Rossi")
        self.assertTrue(fresh(entry, entry["inputHash"], now, 90, False))

    def test_old_override_id_is_validated_then_ignored(self):
        overrides, _ = validate_overrides(
            {"version": 1, "players": {"999": {"skip": True}}}, {"1"}
        )
        self.assertEqual(overrides, {})

    def test_derived_sync_preserves_live_fields(self):
        base = [{"id": "1", "fullName": "Mario Rossi", "photoURL": "https://x/p.png",
                 "photoProviderID": "gazzetta:p"}]
        derived = {"generatedAt": "2026-08-25T10:00:00Z",
                   "players": [{"id": "1", "startingProbability": 95}]}
        changed = sync_derived(base, derived, "2026-08-26T10:00:00Z")
        self.assertTrue(changed)
        self.assertEqual(derived["players"][0]["startingProbability"], 95)
        self.assertEqual(derived["players"][0]["photoURL"], "https://x/p.png")
        self.assertEqual(derived["generatedAt"], "2026-08-26T10:00:00Z")
        self.assertFalse(sync_derived(base, derived, "2026-08-27T10:00:00Z"))
        self.assertEqual(derived["generatedAt"], "2026-08-26T10:00:00Z")

    def test_positive_refresh_reuses_identity_without_wikimedia(self):
        from fetch_player_photos import Response

        class NoWikimedia:
            had_transient = False

            def page_qids(self, _):
                raise AssertionError("identity should come from cache")

            def entities(self, _):
                raise AssertionError("identity should come from cache")

        class ImageClient:
            open_hosts = set()

            def get(self, url, **_):
                return Response(fake_png(), {"Content-Type": "image/png"}, 200, url)

        now = datetime(2026, 8, 26, tzinfo=timezone.utc)
        player = {"id": "1", "name": "Krstovic", "fullName": "Nikola Krstović",
                  "team": "Atalanta", "role": "A"}
        old = cache_entry(
            player, {}, "valid", datetime(2025, 1, 1, tzinfo=timezone.utc),
            fullName="Nikola Krstović", birthDate="2000-04-05", wikidataID="Q28099191",
            photoURL=photo_url("Nikola Krstović", "2000-04-05"),
            photoProviderID="gazzetta:nikola_krstovic_05042000",
        )
        cache = {"version": 1, "players": {"1": old}}
        resolve_players([player], [player], {}, cache, NoWikimedia(), ImageClient(), set(), now, [])
        self.assertEqual(cache["players"]["1"]["status"], "valid")
        self.assertEqual(cache["players"]["1"]["wikidataID"], "Q28099191")


if __name__ == "__main__":
    unittest.main()
