"""Una sola definizione di «il modello», per chi lo misura e per chi lo serve.

Fino a qui la catena che porta dalle medie storiche ai fantapunti esisteva
scritta due volte: in `backtest.py`, che decide se il progetto ha senso, e in
`export_app_data.py`, che produce il file che l'app legge. Le due copie erano
gia' divergenti in tre punti, e la divergenza andava tutta nella stessa
direzione -- **il backtest misurava un modello piu' povero di quello che gira
davvero**:

- `lineups.apply` c'era solo in produzione. E' lo stadio che sostituisce la
  probabilita' di giocare dedotta dalle presenze con quella dichiarata dalle
  probabili formazioni, e che assegna i rigori al rigorista invece di
  spalmarli. Sono i due stadi che spostano di piu'.
- `fantaplayer.enrich_previous` c'era solo in produzione: allunga il passato
  di chi la stagione scorsa in Serie A non c'era.
- La difficolta' della partita si tarava sulle stagioni di taratura nel
  backtest e su tutte in produzione -- differenza corretta, ma implicita.

Due catene che calcolano la stessa cosa divergono sempre, e la divergenza non
da' nessun sintomo: il backtest continua a stampare un numero credibile, solo
che quel numero non riguarda piu' il modello che si sta spedendo. Da qui in
avanti la catena e' una, e le differenze legittime fra i due usi sono
argomenti espliciti al posto di righe copiate.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from fantasquama import calibrate, lineups, odds
from fantasquama.estimate import (
    Shrinkage,
    apply_match_context,
    blended_history,
    event_probabilities,
)
from fantasquama.fixtures import Difficulty, team_factors

# Con le quote in mano il profilo di squadra e' quasi tutto ridondante: le
# quote lo contengono gia', misurato meglio. L'esponente lo riduce a una
# rifinitura invece di lasciarlo contare una seconda volta.
SQUAD_REFINEMENT: float = 0.20


class Factors(NamedTuple):
    """I fattori di difficolta' di ogni riga, e la sola parte di mercato.

    L'app riceve entrambi: `attack`/`defense` sono quelli applicati, i due
    `market_*` sono la componente che viene dalle quote. Tenerli separati
    permette di mostrare quanto di una correzione e' mercato e quanto e'
    profilo di squadra, invece di un numero solo che non si sa da dove venga.
    """

    attack: np.ndarray
    defense: np.ndarray
    market_attack: np.ndarray
    market_defense: np.ndarray
    has_market: np.ndarray


def match_factors(
    context: pd.DataFrame,
    difficulty: Difficulty,
    lambdas: pd.DataFrame | None = None,
    league_goals: float = odds.LEAGUE_GOALS,
) -> Factors:
    """Fattore offensivo e difensivo di ogni riga, quote prima di tutto.

    Tre livelli, in ordine di qualita' e non mescolati fra loro -- e' la
    gerarchia che sostituisce il vecchio impasto fra quote e tassi empirici:

    1. **I gol attesi ricavati dal mercato** (`lambdas`), quando ci sono. Sono
       una risposta diretta alla domanda «quanti gol segna e subisce questa
       squadra in questa partita», e il fattore e' semplicemente quel numero
       rapportato alla media del campionato.
    2. **La retta sul vantaggio di mercato** (`difficulty`), che usa le sole
       quote 1X2 e le riassume in un numero: era il livello 1 di prima.
    3. **Il profilo di squadra** (`team_factors`), che non usa le quote
       affatto e serve dove non ce ne sono.

    Sommare un'informazione peggiore a una migliore non la migliora: le quote
    contengono gia' forma, infortuni e classifica, misurati meglio di come li
    misuri tu. Per questo il profilo di squadra rifinisce appena (esponente
    0,20) dove il mercato c'e', e prende il posto per intero dove manca.
    """
    advantage = (context["p_win"] - context["p_lose"]).to_numpy(np.float64)
    market_attack, market_defense = difficulty.factors(advantage)
    squad_attack, squad_defense = team_factors(context)
    has_market = np.isfinite(advantage)

    if lambdas is not None and league_goals > 0:
        lambda_for = pd.to_numeric(lambdas["lambda_for"], errors="coerce").to_numpy(np.float64)
        lambda_against = pd.to_numeric(lambdas["lambda_against"], errors="coerce").to_numpy(np.float64)
        quotate = np.isfinite(lambda_for) & np.isfinite(lambda_against)
        market_attack = np.where(quotate, lambda_for / league_goals, market_attack)
        market_defense = np.where(quotate, lambda_against / league_goals, market_defense)
        has_market = has_market | quotate

    return Factors(
        attack=market_attack * np.where(has_market, squad_attack**SQUAD_REFINEMENT, squad_attack),
        defense=market_defense * np.where(has_market, squad_defense**SQUAD_REFINEMENT, squad_defense),
        market_attack=market_attack,
        market_defense=market_defense,
        has_market=has_market,
    )


class Estimate(NamedTuple):
    """Quello che il modello sa dire di ogni riga, prima del regolamento.

    `blended` non e' un sottoprodotto: e' la matrice delle medie storiche
    gia' attenuate verso la stagione precedente, ed e' l'ingresso corretto
    sia del calibratore sia del modello addestrato. Viene restituita perche'
    passarne una diversa ai due era gia' successo -- la produzione dava al
    modello addestrato la storia grezza e il confronto le medie attenuate,
    cioe' misurava un modello e ne spediva un altro.
    """

    probabilities: pd.DataFrame
    votes: pd.Series
    blended: pd.DataFrame
    lambdas: pd.DataFrame | None


def estimate(
    archive: pd.DataFrame,
    history: pd.DataFrame,
    previous: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    context: pd.DataFrame | None = None,
    difficulty: Difficulty | None = None,
    formazione: pd.DataFrame | None = None,
    market: odds.Market | None = None,
    shrinkage: Shrinkage | None = None,
) -> Estimate:
    """Probabilita' di evento e voto stimato, per ogni riga dell'archivio.

    Ritorna quello che serve a `score_probabilities`, non i punti: i punti
    dipendono dal regolamento della lega, che e' una scelta di chi chiama.

    `train_mask` seleziona le righe da cui il modello puo' imparare -- prior
    dello shrinkage e taratura del voto. E' obbligatorio e senza default per
    lo stesso motivo per cui lo e' in `event_probabilities`: passare "tutte le
    righe" su un frame che comprende le giornate da prevedere fa leggere al
    modello il proprio futuro, e il rapporto che ne esce e' credibile e
    sbagliato.

    `previous` arriva gia' pronto dal chiamante invece di essere calcolato
    qui: la produzione lo allunga con lo storico FantaPlayer, il backtest no,
    ed e' una differenza che va vista al punto di chiamata.

    `formazione` sono le colonne delle probabili (`slot`, `titolarita`,
    `rigori`). Senza, gli stadi che ne dipendono non girano e la stima resta
    quella dedotta dalle presenze: e' cio' che succede su ogni giornata
    passata, perche' le probabili di allora non le ha conservate nessuno --
    vedi `archivia_probabili` in `aggiorna.py`.

    `context` e `difficulty` vanno insieme: la difficolta' e' una funzione del
    vantaggio di mercato, che sta nel contesto. Uno senza l'altro sarebbe una
    correzione a meta', quindi e' un errore invece che un silenzio.
    """
    if (context is None) != (difficulty is None):
        raise ValueError(
            "context e difficulty vanno passati insieme: la difficolta' si legge "
            "dal vantaggio di mercato, che sta nel contesto"
        )

    blended = blended_history(history, previous)
    lambdas = odds.align(market, archive) if market is not None else None
    media = market.league_goals if market is not None else odds.LEAGUE_GOALS

    # Il voto stimato guarda anche la partita, non solo la storia: e' il
    # predittore piu' forte che ci sia, vedi MATCH_FEATURES. Dove le quote
    # mancano si mette la media del campionato, cioe' una partita
    # equilibrata: e' il valore neutro, e lascia la riga utilizzabile invece
    # di farle perdere anche tutte le altre feature -- `calibrate.predict`
    # scarta la riga intera appena una colonna e' NaN.
    per_voto = blended.assign(
        lambda_for=media if lambdas is None else lambdas["lambda_for"].fillna(media).to_numpy(),
        lambda_against=media if lambdas is None else lambdas["lambda_against"].fillna(media).to_numpy(),
    )
    models = calibrate.fit(
        per_voto.iloc[train_mask],
        archive["role"].iloc[train_mask],
        archive["voto"].iloc[train_mask],
    )
    votes = calibrate.predict(models, per_voto, archive["role"])
    probabilities = event_probabilities(
        history, archive["role"], train_mask, previous, shrinkage=shrinkage
    )

    # Le formazioni entrano PRIMA della difficolta' della partita: i rigori
    # che assegnano sono quelli di una squadra media, e una squadra favorita
    # ne ottiene di piu'. Dopo, la correzione non li toccherebbe.
    if formazione is not None:
        probabilities = lineups.apply(
            probabilities, archive["role"], archive["team"],
            formazione["slot"], formazione["titolarita"], formazione["rigori"],
        )

    if context is not None and difficulty is not None:
        fattori = match_factors(context, difficulty, lambdas, media)
        probabilities = apply_match_context(probabilities, fattori.attack, fattori.defense)

        # I gol subiti e la porta inviolata del portiere il mercato li prezza
        # quasi esattamente, quindi li sostituisce invece di scalarli: vale
        # DOPO la correzione di difficolta', altrimenti quest'ultima li
        # scalerebbe una seconda volta.
        # I gol si spartiscono DOPO la correzione di difficolta': quella
        # stabilisce le proporzioni fra i giocatori, questa fissa il totale
        # della squadra. Invertirle rimetterebbe il doppio conteggio.
        if market is not None:
            probabilities = odds.allocate_attack(
                probabilities, archive, lambdas["lambda_for"].to_numpy(np.float64)
            )
            probabilities = odds.apply_goalkeeper(
                probabilities, archive["role"],
                lambdas["lambda_against"].to_numpy(np.float64), market,
            )

    return Estimate(
        probabilities=probabilities, votes=votes, blended=blended, lambdas=lambdas
    )
