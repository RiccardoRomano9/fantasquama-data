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
    """Bonus e malus della lega. I default sono il regolamento Classic.

    **`sv` e' il valore del SOSTITUTO, non del giocatore assente.** Chi non
    prende voto in una lega vera non lascia un buco: entra un panchinaro al
    suo posto, e quel panchinaro un voto lo prende -- intorno al 6. Il
    default e' quindi 6.0, cioe' il valore atteso della casella, non zero.

    Con `sv = 0.0` il punteggio atteso diventa `p_vote * fantavoto`, dove
    `fantavoto` sta intorno a 6: il primo fattore domina tutto il resto.

        giocatore   p_vote   fantavoto   punti attesi
        A            0,95        6,0         5,70
        B            0,75        7,2         5,40

    A vince pur valendo 1,2 fantavoto in meno quando gioca; perche' B lo
    superi servirebbe `fv > 6,0 x 0,95/0,75 = 7,6`. Un divario di 0,2 in
    `p_vote` chiede 1,6 fantavoto per essere compensato, cosa che non
    succede quasi mai: **il ranking diventa una copia di `p_vote`**. E le
    baseline del backtest ("chi ha giocato di piu'") sono due stimatori
    grezzi proprio di `p_vote`, quindi il modello finirebbe a competere con
    una versione rumorosa del proprio termine dominante. Misurato: con
    `sv = 0` fra baseline e soffitto ci sono 4,3 punti in tutto, con
    `sv = 6` ce ne sono 21,5.

    `sv = 0.0` resta disponibile, ed e' la scelta giusta per le leghe senza
    sostituzioni, dove la casella vuota vale davvero zero. Non e' pero' il
    caso comune, e non deve essere il default.
    """

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
    sv: float = 6.0    # valore del sostituto che entra al posto di chi non prende voto


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
