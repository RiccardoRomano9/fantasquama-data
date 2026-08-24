"""Funzione punteggio: eventi per regole della lega, uguale fantapunti.

Pura e deterministica. Funziona sia con conteggi interi, per ricostruire un
fantavoto realmente avvenuto, sia con probabilita', per stimare quello atteso:
e' la stessa somma pesata in entrambi i casi.

Questo modulo verra' riscritto in Swift tale e quale. Niente pandas, niente
numpy, niente stato.
"""

from collections.abc import Mapping
from dataclasses import dataclass

EVENTS: tuple[str, ...] = ("gf", "rf", "rs", "rp", "gs", "au", "amm", "esp", "ass", "cs")


@dataclass(frozen=True)
class Rules:
    """Bonus e malus della lega. I default sono il regolamento Classic."""

    gf: float = 3.0    # gol su azione
    rf: float = 3.0    # rigore segnato
    rs: float = -3.0   # rigore sbagliato
    rp: float = 3.0    # rigore parato, solo portieri
    gs: float = -1.0   # gol subito, solo portieri
    au: float = -2.0   # autogol
    amm: float = -0.5  # ammonizione
    esp: float = -1.0  # espulsione
    ass: float = 1.0   # assist
    cs: float = 1.0    # porta inviolata, solo portieri
    sv: float = 0.0    # valore d'ufficio quando il giocatore non prende voto


def fantavoto(voto: float, events: Mapping[str, float], rules: Rules) -> float:
    """Fantavoto di una prestazione: il voto piu' la somma pesata degli eventi."""
    return voto + sum(getattr(rules, name) * events.get(name, 0.0) for name in EVENTS)


def expected_points(
    voto: float,
    events: Mapping[str, float],
    rules: Rules,
    p_vote: float,
) -> float:
    """Fantapunti attesi: media fra il caso in cui gioca e quello in cui non gioca.

    `events` contiene probabilita', non conteggi. `p_vote` e' la probabilita'
    di scendere in campo e restarci abbastanza da prendere un voto.
    """
    return p_vote * fantavoto(voto, events, rules) + (1.0 - p_vote) * rules.sv
