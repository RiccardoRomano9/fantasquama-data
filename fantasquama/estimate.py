"""Dalla storia alle probabilita' della prossima giornata.

Ogni grandezza e' mescolata con un prior tramite uno shrinkage:

    stima = (n * osservato + k * prior) / (n + k)

Con poche presenze domina il prior, con molte domina il giocatore. Nessuna
soglia, nessun caso limite.

Il prior e' calcolato per ruolo, non sull'intero archivio: un attaccante
segna un ordine di grandezza piu' spesso di un difensore, e il prior e'
proprio cio' che decide la stima quando le presenze sono poche -- cioe' a
inizio stagione, il regime che il backtest deve giudicare con piu' attenzione.
Un prior globale annacquerebbe il modello esattamente li' dove conta di piu':
non va risemplificato a una media su tutti i ruoli insieme.

Il prior si calcola sulle sole righe indicate da `prior_mask` -- nel backtest
le stagioni di taratura -- e poi si applica a tutte le righe. Calcolarlo
sull'intero archivio farebbe entrare nella stima di una giornata i dati delle
giornate successive della stagione da giudicare: una fuga di informazione
misurata, non teorica (0,36 fantapunti di scostamento su una riga il cui
dato non era cambiato). E' la stessa popolazione su cui si tara il
calibratore, quindi il confine fra cio' che il modello ha visto e cio' che
deve prevedere e' uno solo in tutto il backtest.

`prior_mask` e' un argomento OBBLIGATORIO, senza valore di default, e non va
reso opzionale "per comodita'". Un default che significasse "tutte le righe"
rimetterebbe la fuga in silenzio ogni volta che qualcuno lo dimentica: il
backtest girerebbe fino in fondo e stamperebbe un rapporto dall'aria normale,
prodotto da un modello che ha letto il futuro. Senza default, dimenticarlo e'
un TypeError immediato. La differenza fra le due cose e' la differenza fra un
errore che si nota e un numero sbagliato a cui si crede -- ed e' su questo
numero che si decide se il progetto va avanti.

Il prior di ruolo e' pero' l'ultima risorsa, non la prima. Se il giocatore
ha giocato la stagione precedente, il suo prior e' cio' che ha fatto ALLORA,
attenuato verso la media di ruolo in proporzione alle presenze: e' la spec
5.a, ed e' quello che rende sensata la prima giornata, quando di questa
stagione non si sa ancora niente. Vedi `previous_season`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gammaln

from fantasquama.features import ROLLING_WINDOW
from fantasquama.scoring import EVENTS, Rules, expected_points

K: float = 4.0

# Quanto il ritmo offensivo della squadra puo' spostare la probabilita' di
# bonus di un suo giocatore. Un centrocampista dell'Inter e uno del Lecce
# possono avere la stessa storia personale senza avere la stessa occasione:
# senza questa correzione il modello li tratta identici.
#
# ponytail: gli estremi sono a occhio, servono solo a impedire che una squadra
# con due giornate di storia e cinque gol moltiplichi per otto. Da tarare sul
# backtest, come K.
TEAM_FACTOR_MIN: float = 0.5
TEAM_FACTOR_MAX: float = 2.0

# Gli eventi che dipendono da quanto segna la squadra. Cartellini, autoreti e
# le voci da portiere non ci entrano: non c'e' ragione di pensare che una
# squadra che segna di piu' faccia ammonire di piu' i propri centrocampisti.
TEAM_SCALED_EVENTS: tuple[str, ...] = ("gf", "rf", "ass")


def shrink(
    observed: pd.Series,
    prior: pd.Series | float,
    n: pd.Series,
    k: float | np.ndarray | pd.Series = K,
) -> pd.Series:
    """Media pesata fra il valore osservato e il prior, con peso crescente su n.

    `k` puo' essere un numero solo o un valore per riga: serve da quando e'
    stimato per (ruolo, evento) invece di essere una costante -- vedi
    `fit_shrinkage`.
    """
    observed = pd.to_numeric(observed, errors="coerce").fillna(0.0)
    n = pd.to_numeric(n, errors="coerce").fillna(0.0)
    if isinstance(k, pd.Series):
        k = k.to_numpy()
    return (n * observed + k * prior) / (n + k)


# Estremi entro cui puo' finire un k stimato. Servono contro i due modi in cui
# la stima puo' degenerare su un evento quasi mai osservato: un k prossimo a
# zero, che promuoverebbe a certezza il singolo cartellino rosso di una
# carriera, e un k enorme, che renderebbe ogni giocatore identico alla media
# del suo ruolo per sempre.
K_MIN: float = 0.5
K_MAX: float = 500.0

# Il Gamma-Poisson attenua troppo poco, e si vede.
#
# La pendenza di calibrazione -- la logistica di cio' che e' avvenuto su
# logit(previsto), dove 1 significa "il modello si sbilancia quanto deve" --
# per il gol di attaccanti e centrocampisti valeva 0,768, con errore standard
# 0,063: 3,7 deviazioni sotto 1, non e' rumore. Tradotto: al quintile piu'
# basso prometteva il 3,2% di probabilita' di gol e ne usciva il 5,2%, al piu'
# alto prometteva 21,2% e ne usciva 18,7%. Il livello medio era giusto (0,1010
# previsto contro 0,0994 reale), la forma no -- ed e' la forma che ordina i
# giocatori fra loro, cioe' l'unica cosa che il modello serva a fare.
#
# IL MECCANISMO, per quanto ne ho verificato. La verosimiglianza binomiale
# negativa vede solo i totali di stagione, e li spiega con due sorgenti di
# varianza: la Poisson dentro il giocatore e la Gamma fra giocatori. Se dentro
# il giocatore c'e' piu' varianza di quanta una Poisson ne preveda, quella in
# piu' non ha dove andare se non nella Gamma: il modello conclude che i
# giocatori differiscano fra loro piu' di quanto facciano, alfa esce piccolo,
# beta con lui, e beta e' proprio `k`. Riprodotto in laboratorio -- vedi
# `test_la_sovradispersione_dentro_il_giocatore_abbassa_il_k`: a parita' di
# tassi veri, con sovradispersione il k stimato scende da 14,6 a 12,9.
#
# QUELLO CHE IL MECCANISMO NON SPIEGA. Quell'effetto vale l'11%, la correzione
# che serve e' il 100%. Il prior personale -- attenuare verso la stagione
# scorsa invece che verso la media di ruolo -- ne spiega un altro pezzo:
# togliendolo la pendenza sale da 0,768 a 0,815, ma il Brier peggiora, quindi
# resta. Il resto non l'ho isolato. Il fattore qui sotto e' quindi una
# costante TARATA, non la conseguenza di un conto: e' onesto chiamarla cosi'.
#
# COME E' STATO SCELTO IL 2. Su una validazione interna -- taratura 2023-24,
# scelta su 2024-25 -- senza mai guardare la stagione di verifica. Due criteri
# indipendenti danno lo stesso numero:
#
#     k x     pendenza (2024-25)   accuratezza decisionale
#     1.0     0.880  (z = -1.96)               0.6287
#     1.5     0.955  (z = -0.69)               0.6301
#     2.0     1.001  (z = +0.02)               0.6328   <-
#     3.0     1.055  (z = +0.76)               0.6322
#
# Vale per tutti gli eventi e non solo per il gol: il difetto e' della
# verosimiglianza, non di una colonna. Sugli altri la pendenza era gia' entro
# il rumore di 1 (assist 0,90, ammonizione 0,90, porta inviolata 0,96) e il
# fattore la lascia li'.
#
# QUELLO CHE RESTA. Anche col fattore la pendenza si ferma a 0,864. Il pezzo
# mancante e' `p_vote`, che e' troppo sicuro dei titolari (promette 96,9%, ne
# giocano il 90,5%) e percio' gonfia la quota dei migliori nella spartizione
# dei gol. Misurato con un oracolo -- rimpiazzando p_vote con chi ha giocato
# davvero -- la pendenza arriva a 0,917, cioe' dentro il rumore di 1. Non e'
# aggiustabile da qui: serve lo storico delle probabili formazioni, che
# `probabili-archivio/` sta accumulando una giornata alla volta.
#
# ponytail: se un domani la pendenza tornasse lontana da 1, si stima la
# sovradispersione con un terzo parametro invece di ritoccare questo numero.
K_INFLATION: float = 2.0


@dataclass(frozen=True)
class Shrinkage:
    """Quanto fidarsi del giocatore invece che della media, evento per evento.

    Con un `k` unico per tutti gli eventi -- 4, scelto a occhio -- lo stesso
    peso valeva per il gol di un attaccante (che ne fa 0,35 a presenza) e per
    l'espulsione (0,003). Ma quanto conviene fidarsi del dato personale
    dipende dal rapporto fra la varianza *fra* giocatori e quella *dentro* lo
    stesso giocatore, e quel rapporto cambia di ordini di grandezza fra un
    evento e l'altro: dopo quattro presenze il tasso di ammonizione di un
    singolo e' quasi tutto rumore, e con k = 4 quel rumore contava gia' come
    meta' della stima.

    `k` e prior si stimano per (ruolo, evento) con un Gamma-Poisson: gli
    eventi sono conteggi e non esiti binari -- un attaccante puo' segnare due
    gol nella stessa partita -- quindi il coniugato giusto e' la Gamma, non
    la Beta. La media a posteriori che ne esce e' esattamente
    `(n*osservato + k*prior)/(n + k)` con `k = beta` e `prior = alfa/beta`,
    cioe' la stessa formula gia' in uso: cambia solo da dove viene il numero.
    """

    per_event: dict[tuple[str, str], tuple[float, float]]
    default_k: float = K

    def k_for(self, roles: np.ndarray, event: str) -> np.ndarray:
        """Il k di ogni riga, dato il suo ruolo."""
        return np.array(
            [self.per_event.get((r, event), (self.default_k, np.nan))[0] for r in roles],
            dtype=np.float64,
        )

    def prior_for(self, roles: np.ndarray, event: str) -> np.ndarray:
        """Il prior di ruolo stimato, NaN dove non c'era abbastanza per stimarlo."""
        return np.array(
            [self.per_event.get((r, event), (self.default_k, np.nan))[1] for r in roles],
            dtype=np.float64,
        )


def _negative_binomial_loglik(parameters: np.ndarray, totals: np.ndarray, exposure: np.ndarray) -> float:
    """Verosimiglianza marginale di un Gamma-Poisson, cambiata di segno.

    Integrando la Poisson sul prior Gamma(alfa, beta) resta una binomiale
    negativa, che dipende solo da (alfa, beta) e dai dati: e' il modo di
    stimare il prior senza dover stimare anche i tassi dei singoli.
    """
    alpha, beta = np.exp(parameters)  # esponenziale: restano positivi
    return -float(
        (
            gammaln(totals + alpha) - gammaln(alpha) - gammaln(totals + 1.0)
            + alpha * np.log(beta / (beta + exposure))
            + totals * np.log(exposure / (beta + exposure))
        ).sum()
    )


def fit_shrinkage(
    archive: pd.DataFrame, mask: np.ndarray, events: tuple[str, ...] = EVENTS
) -> Shrinkage:
    """Stima `k` e prior per ogni (ruolo, evento) sulle sole righe di `mask`.

    `mask` e' la stessa popolazione di taratura di tutto il resto: stimare
    quanto fidarsi del dato personale guardando anche le giornate da
    prevedere sarebbe una fuga di informazione come le altre.

    L'unita' di osservazione e' il giocatore-stagione, non la riga: quello che
    si vuole misurare e' quanto i giocatori differiscono fra loro rispetto a
    quanto oscillano da soli, e per vederlo serve un totale per giocatore.
    """
    dati = archive.loc[mask]
    giocate = dati["played"].to_numpy(bool)
    per_evento: dict[tuple[str, str], tuple[float, float]] = {}

    chiavi = pd.DataFrame({
        "role": dati["role"].fillna("").astype(str).to_numpy(),
        "player_id": dati["player_id"].astype(str).to_numpy(),
        "season": dati["season"].astype(str).to_numpy(),
        "apps": giocate.astype(np.float64),
    })

    for evento in events:
        valori = pd.to_numeric(dati[evento], errors="coerce").fillna(0.0).to_numpy() * giocate
        frame = chiavi.assign(total=valori)
        per_giocatore = frame.groupby(["role", "player_id", "season"], observed=True).agg(
            total=("total", "sum"), apps=("apps", "sum")
        ).reset_index()
        per_giocatore = per_giocatore[per_giocatore["apps"] > 0]

        for role, gruppo in per_giocatore.groupby("role", observed=True):
            totals = gruppo["total"].to_numpy(np.float64)
            exposure = gruppo["apps"].to_numpy(np.float64)
            # Serve almeno un evento osservato: su una colonna tutta a zero la
            # verosimiglianza e' piatta in alfa e l'ottimizzazione restituisce
            # il punto di partenza travestito da stima.
            if len(totals) < 30 or totals.sum() <= 0:
                continue

            media = max(totals.sum() / exposure.sum(), 1e-6)
            esito = optimize.minimize(
                _negative_binomial_loglik,
                x0=np.log([media * K, K]),  # alfa/beta = media, beta = K
                args=(totals, exposure),
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 500},
            )
            alpha, beta = np.exp(esito.x)
            if not (np.isfinite(alpha) and np.isfinite(beta)):
                continue
            # il gonfiaggio va sul solo k: il prior alfa/beta e' il livello
            # medio del ruolo, ed era gia' giusto -- vedi K_INFLATION
            k = float(np.clip(beta * K_INFLATION, K_MIN, K_MAX))
            per_evento[(str(role), evento)] = (k, float(alpha / beta))

    return Shrinkage(per_event=per_evento)


def _role_prior(
    column: pd.Series, role_values: np.ndarray, default: float, prior_mask: np.ndarray
) -> pd.Series:
    """Media della colonna dentro ciascun ruolo, con ricaduta sulla media generale.

    La media si calcola sulle sole righe di `prior_mask` e si applica a tutte
    le righe di quel ruolo: le righe fuori dalla maschera ricevono un prior,
    non lo producono.

    Lavora su array numpy, non su un raggruppamento per etichetta di indice:
    un indice non unico (possibile a valle di `rolling_history`) non deve
    poter disallineare il risultato, lo stesso motivo per cui
    `rolling_history` e `calibrate.fit` mascherano posizionalmente anziche'
    per indice. Il risultato porta l'indice di `column`, cosi' che `shrink`
    si allinei correttamente.
    """
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64)
    prior = np.full(len(values), default, dtype=np.float64)
    for role in np.unique(role_values):
        mask = role_values == role
        valid = values[mask & prior_mask]
        valid = valid[~np.isnan(valid)]
        if valid.size == 0:
            continue  # nessun dato per questo ruolo: resta il default (la media generale)
        prior[mask] = valid.mean()
    return pd.Series(prior, index=column.index)


# `voto` sta accanto agli eventi: anche il voto base e' una cosa che il
# giocatore si porta dietro dall'anno prima.
PREVIOUS_STATS: tuple[str, ...] = ("voto", *EVENTS)

PREVIOUS_COLUMNS: tuple[str, ...] = (
    *(f"{name}_prev" for name in PREVIOUS_STATS),
    "vote_rate_prev",
    "apps_prev",
)


def _previous_label(season: object) -> str:
    """`"2026-27"` -> `"2025-26"`.

    Si ricava dall'etichetta e non dall'elenco delle stagioni presenti, cosi'
    funziona anche per una stagione che in archivio non c'e' ancora -- il caso
    per cui serve di piu': la 2026-27, di cui non esiste una sola giornata.
    """
    year = int(str(season)[:4])
    return f"{year - 1}-{year % 100:02d}"


def previous_season(archive: pd.DataFrame) -> pd.DataFrame:
    """Come ha giocato ogni giocatore la stagione PRIMA di quella della riga.

    Ritorna un frame allineato per posizione ad `archive`, con la media di
    ogni evento sulle giornate in cui ha preso voto, la quota di giornate in
    cui l'ha preso, e quante presenze ha messo insieme. Chi la stagione prima
    non c'era ha tutto NaN, e a valle riceve il solo prior di ruolo.

    Non serve nessuna maschera: al momento di prevedere una giornata della
    stagione S, la stagione S-1 e' passato per intero. E' proprio la
    differenza fra questo e le aggregate della stagione corrente, che invece
    conterrebbero le giornate ancora da giocare.

    L'aggregato si calcola sulla stagione intera anche per chi ha cambiato
    squadra: quello che porta con se' e' come ha giocato, non dove.
    """
    played = archive["played"].astype(bool).to_numpy()
    frame = pd.DataFrame({
        "season": archive["season"].astype(str).to_numpy(),
        "player_id": archive["player_id"].astype(str).to_numpy(),
        "played": played.astype(np.float64),
        **{
            name: (pd.to_numeric(archive[name], errors="coerce").to_numpy(np.float64) * played)
            for name in PREVIOUS_STATS
        },
    })
    key = ["season", "player_id"]
    grouped = frame.groupby(key, observed=True)
    apps = grouped["played"].sum()
    # il denominatore e' quante giornate quel giocatore ha attraversato, non
    # quante ne ha la stagione: chi arriva a gennaio non deve risultare uno
    # che ha saltato mezzo campionato. E' la stessa definizione di
    # `gw_elapsed` in rolling_history.
    elapsed = grouped.size()

    aggregate = grouped[list(PREVIOUS_STATS)].sum().div(apps, axis=0)
    aggregate.columns = [f"{name}_prev" for name in PREVIOUS_STATS]
    aggregate["apps_prev"] = apps
    aggregate["vote_rate_prev"] = apps / elapsed
    aggregate = aggregate.reset_index()

    wanted = pd.DataFrame({
        "season": [_previous_label(s) for s in frame["season"]],
        "player_id": frame["player_id"],
    })
    merged = wanted.merge(aggregate, on=key, how="left")
    # il merge azzera l'indice: si rimette per posizione, come ovunque
    return pd.DataFrame(
        merged[list(PREVIOUS_COLUMNS)].to_numpy(np.float64),
        columns=list(PREVIOUS_COLUMNS),
        index=archive.index,
    )


def _blend(
    prior: pd.Series, previous: pd.DataFrame | None, column: str,
    k: float | np.ndarray = K,
) -> pd.Series:
    """Attenua il prior di ruolo verso quello che il giocatore ha fatto l'anno prima.

    Chi ha una stagione intera alle spalle si porta dietro quasi solo se
    stesso; chi ne ha due giornate resta quasi tutto media di ruolo; chi non
    c'era riceve la media di ruolo e basta -- `shrink` con n = 0 restituisce
    esattamente il prior.
    """
    if previous is None:
        return prior
    if len(previous) != len(prior):
        raise ValueError("previous deve avere una riga per ogni riga di history")
    observed = pd.Series(previous[column].to_numpy(), index=prior.index)
    n = pd.Series(previous["apps_prev"].to_numpy(), index=prior.index)
    return shrink(observed, prior, n, k)


def blended_history(history: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """`history` con le medie attenuate verso la stagione scorsa.

    `event_probabilities` questo lo fa gia' per conto suo, dentro lo shrinkage.
    Serve a chi le medie le legge dritte -- il calibratore del voto, che non
    passa di li'. Senza, alla prima giornata legge una colonna di zeri e stima
    lo stesso voto per tutti: 6,04 dal primo attaccante all'ultimo portiere.

    Chi la stagione scorsa non c'era tiene i propri valori: `shrink` con un
    prior uguale all'osservato restituisce l'osservato.
    """
    if len(previous) != len(history):
        raise ValueError("previous deve avere una riga per ogni riga di history")
    out = history.copy()
    n = pd.to_numeric(history["apps_before"], errors="coerce").fillna(0.0)
    for name in PREVIOUS_STATS:
        column = pd.to_numeric(history[f"{name}_mean"], errors="coerce")
        prior = pd.Series(previous[f"{name}_prev"].to_numpy(), index=history.index)
        out[f"{name}_mean"] = shrink(column, prior.fillna(column), n)
    return out


def _prior_mask(history: pd.DataFrame, prior_mask: np.ndarray) -> np.ndarray:
    """Normalizza la maschera del prior e ne verifica la lunghezza.

    Non c'e' un valore di default, di proposito: vedi `event_probabilities`.
    """
    mask = np.asarray(prior_mask, dtype=bool)
    if len(mask) != len(history):
        raise ValueError("prior_mask deve avere una riga per ogni riga di history")
    if not mask.any():
        raise ValueError("prior_mask non seleziona nessuna riga: nessun prior calcolabile")
    return mask


def _team_factor(history: pd.DataFrame, prior_mask: np.ndarray) -> np.ndarray:
    """Quanto la squadra segna rispetto alla media, per riga.

    La media di riferimento viene dalle sole righe di `prior_mask`, come i
    prior: prenderla su tutto il frame farebbe entrare la stagione di verifica
    nel calcolo di una grandezza usata per predirla.

    Una squadra senza storia (prima giornata) ha fattore 1: nessuna
    correzione, non una correzione verso zero.
    """
    rates = pd.to_numeric(history["team_goals_rate"], errors="coerce").to_numpy(np.float64)
    riferimento = rates[prior_mask]
    riferimento = riferimento[~np.isnan(riferimento)]
    if riferimento.size == 0 or riferimento.mean() <= 0:
        return np.ones(len(history))

    factor = rates / riferimento.mean()
    factor[np.isnan(factor)] = 1.0
    return np.clip(factor, TEAM_FACTOR_MIN, TEAM_FACTOR_MAX)


def event_probabilities(
    history: pd.DataFrame,
    roles: pd.Series,
    prior_mask: np.ndarray,
    previous: pd.DataFrame | None = None,
    k: float = K,
    shrinkage: Shrinkage | None = None,
) -> pd.DataFrame:
    """Probabilita' per evento e probabilita' di prendere voto, con prior per ruolo.

    `prior_mask` seleziona le righe da cui i prior POSSONO essere calcolati.
    Le altre righe un prior lo ricevono soltanto, non lo producono.

    `previous` e' l'uscita di `previous_season`, allineata per posizione: con
    essa il prior di ogni riga non e' piu' la media del ruolo ma quello che
    quel giocatore ha fatto la stagione prima, attenuato verso la media di
    ruolo in proporzione alle sue presenze. Ometterlo non introduce nessuna
    fuga -- si torna al prior di ruolo, che e' solo piu' povero -- quindi qui
    un default c'e', a differenza di `prior_mask`.

    L'argomento e' obbligatorio e non ha un default. Passare "tutte le righe"
    quando il frame comprende la stagione di verifica rimette esattamente la
    fuga di informazione che questa maschera esiste per togliere: la stima di
    una giornata tornerebbe a leggere le giornate successive della stagione
    che deve prevedere, e il backtest misurerebbe un modello che ha visto il
    futuro senza che nulla, nel rapporto, lo lasci capire. Un default avrebbe
    reso quel ritorno silenzioso; l'assenza di default lo rende un TypeError.
    Nel backtest la maschera e' quella delle stagioni di taratura, la stessa
    su cui si tara il calibratore.
    """
    mask = _prior_mask(history, prior_mask)
    n = pd.to_numeric(history["apps_before"], errors="coerce").fillna(0.0)
    # fillna("") prima del confronto per ruolo: roles puo' avere dtype nullable
    # "string" (come lo produce ingest.py). Un ruolo mancante forma cosi' un
    # proprio gruppo (vuoto di dati), che ricade sulla media generale invece
    # di mischiarsi in un ruolo vero o far esplodere il confronto.
    role_values = roles.fillna("").astype(str).to_numpy()
    out = pd.DataFrame(index=history.index)

    for name in EVENTS:
        column = history[f"{name}_mean"]
        # controllo esplicito, non `x or 0.0`: NaN e' truthy in Python, quindi
        # con una media NaN l'idioma `or` restituirebbe NaN, non il default.
        # anche la media generale, che fa da ricaduta, guarda solo le righe
        # di `mask`: altrimenti la fuga rientrerebbe dalla porta di servizio
        mean_value = pd.to_numeric(column, errors="coerce")[mask].mean()
        default = 0.0 if pd.isna(mean_value) else float(mean_value)
        prior = _role_prior(column, role_values, default, mask)
        # Il peso del prior e' stimato per (ruolo, evento) quando `shrinkage`
        # c'e': con un k unico, il tasso di espulsione di chi ha quattro
        # presenze contava gia' per meta' della stima. Vedi `fit_shrinkage`.
        peso = shrinkage.k_for(role_values, name) if shrinkage is not None else k
        out[name] = shrink(column, _blend(prior, previous, f"{name}_prev", peso), n, peso)

    # p_vote si basa sulle giornate trascorse, non sulle presenze:
    # chi non e' mai sceso in campo ha un'informazione, non un dato mancante.
    out[list(TEAM_SCALED_EVENTS)] = out[list(TEAM_SCALED_EVENTS)].to_numpy() * _team_factor(
        history, prior_mask
    )[:, None]

    # p_vote e' una catena di ripieghi, dal piu' recente al piu' generico:
    # le ultime cinque giornate, poi la stagione in corso, poi la stagione
    # scorsa, poi la media di ruolo. Ogni anello pesa quanto sono i dati che
    # lo sostengono, e chi non ha il primo cade sul secondo senza casi limite.
    #
    # La recenza NON e' un dettaglio: chi non prende voto vale `rules.sv`, e
    # con un sv basso la domanda «gioca?» pesa piu' di ogni bonus. Un
    # cumulato di stagione dice ancora titolare di chi ha giocato le prime
    # cinque giornate e poi si e' fatto male.
    elapsed = pd.to_numeric(history["gw_elapsed"], errors="coerce").fillna(0.0)
    rate = pd.to_numeric(history["vote_rate"], errors="coerce")
    mean_rate = rate[mask].mean()
    default_rate = 0.5 if pd.isna(mean_rate) else float(mean_rate)
    prior_rate = _role_prior(rate, role_values, default_rate, mask)
    stagione = shrink(rate, _blend(prior_rate, previous, "vote_rate_prev"), elapsed, k)

    recent_n = np.minimum(elapsed.to_numpy(np.float64), ROLLING_WINDOW)
    recent = pd.to_numeric(history["apps_last5"], errors="coerce") / pd.Series(
        np.where(recent_n > 0, recent_n, np.nan), index=history.index
    )
    out["p_vote"] = shrink(recent, stagione,
                           pd.Series(recent_n, index=history.index), k).clip(0.0, 1.0)

    # Qui c'era un `.replace([inf, -inf], nan).fillna(0.0)`. E' esattamente
    # cio' che una volta ha trasformato un prior NaN su `vote_rate` in un
    # p_vote uniformemente 0.0 senza nessun sintomo (vedi il test
    # test_prior_nan_su_vote_rate_non_azzera_p_vote). Oggi non passa mai
    # nessun valore non finito, il che lo rende puro rischio: se un domani un
    # prior tornasse NaN, ogni punteggio collasserebbe su rules.sv e il
    # cancello del progetto fallirebbe in silenzio per il motivo sbagliato.
    # `raise` e non `assert`: con python -O un assert sparisce.
    values = out.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        bad = sorted(out.columns[~finite.all(axis=0)])
        raise ValueError(
            f"probabilita' non finite nelle colonne {bad}: "
            "un prior o una media e' NaN o infinito, e mascherarlo con uno "
            "zero darebbe un backtest sbagliato senza sintomi"
        )
    return out


def expected_fantapoints(
    history: pd.DataFrame,
    roles: pd.Series,
    votes: pd.Series,
    rules: Rules,
    prior_mask: np.ndarray,
    previous: pd.DataFrame | None = None,
) -> pd.Series:
    """Fantapunti attesi per ogni riga, con le regole della lega date.

    `history`, `roles` e `votes` sono paralleli per posizione: la riga i-esima
    di ciascuno descrive la stessa osservazione, e il ciclo piu' sotto le
    accoppia con `.iloc[i]`. Lunghezze diverse non darebbero un errore ma un
    accoppiamento sbagliato, quindi il controllo e' esplicito, come in
    `calibrate.fit`.

    `prior_mask` e' obbligatorio e viene passato pari pari a
    `event_probabilities`: seleziona le righe da cui i prior possono essere
    calcolati. Passare "tutte le righe" su un frame che comprende la stagione
    di verifica rimette la fuga di informazione, e i punteggi che ne escono
    sono quelli di un modello che ha letto il futuro. Nessun default, cosi'
    dimenticarlo e' un TypeError e non un numero sbagliato credibile.
    """
    if not len(history) == len(roles) == len(votes):
        raise ValueError("history, roles e votes devono avere lo stesso numero di righe")

    probabilities = event_probabilities(history, roles, prior_mask, previous)
    return score_probabilities(probabilities, roles, votes, rules)


def score_probabilities(
    probabilities: pd.DataFrame,
    roles: pd.Series,
    votes: pd.Series,
    rules: Rules,
) -> pd.Series:
    """Fantapunti attesi da probabilita' gia' calcolate.

    Separata da `expected_fantapoints` perche' fra le probabilita' e il
    punteggio si inserisce la difficolta' della partita: chi vuole applicarla
    chiama `event_probabilities`, poi `apply_match_context`, poi questa.
    """
    if not len(probabilities) == len(roles) == len(votes):
        raise ValueError("probabilita', roles e votes devono avere lo stesso numero di righe")

    events = probabilities[list(EVENTS)]

    # i bonus da portiere non si applicano agli altri ruoli. fillna("") prima
    # del confronto: roles puo' avere dtype nullable "string" (come lo produce
    # ingest.py), dove `Series != valore` su una riga NA da' pd.NA anziche'
    # False, e usarlo come maschera booleana solleva un errore a valle. Un
    # ruolo mancante viene trattato come esterno: e' la scelta prudente,
    # azzera i bonus da portiere invece di concederli senza fondamento.
    role_values = roles.fillna("").astype(str).to_numpy()
    outfield = role_values != "P"
    events = events.copy()
    events.loc[outfield, ["rp", "gs"]] = 0.0

    return pd.Series(
        [
            expected_points(
                float(votes.iloc[i]),
                events.iloc[i].to_dict(),
                rules,
                p_vote=float(probabilities["p_vote"].iloc[i]),
            )
            for i in range(len(probabilities))
        ],
        index=probabilities.index,
        dtype="float64",
    )


def apply_match_context(
    probabilities: pd.DataFrame,
    attack: np.ndarray,
    defense: np.ndarray,
) -> pd.DataFrame:
    """Scala le probabilita' con la difficolta' della partita.

    Gli eventi offensivi seguono il fattore d'attacco, i gol subiti quello
    difensivo. Cartellini, autoreti e rigori parati non vengono toccati: non
    c'e' ragione di pensare che affrontare una squadra forte faccia ammonire
    di piu' -- e inventare una correzione dove non si ha motivo di aspettarsela
    aggiunge rumore, non segnale.

    **`cs` non si scala moltiplicando.** Gli altri eventi sono conteggi
    attesi, e raddoppiare la difficolta' raddoppia il conteggio; la porta
    inviolata e' invece una probabilita', e una probabilita' moltiplicata per
    due esce dall'intervallo in cui vive. La relazione giusta e' quella che
    lega le due cose: con i gol subiti distribuiti come una Poisson di media
    lambda, `P(cs) = e^-lambda`, quindi se lambda viene scalata di `d` la
    porta inviolata diventa `e^(-lambda*d)`, cioe' `P(cs)` elevato a `d`.
    Una partita due volte piu' dura porta la probabilita' da 0,30 a 0,09;
    una due volte piu' facile la porta a 0,55. Resta in [0, 1] da sola,
    senza nessun tetto arbitrario.

    Prima questa riga non c'era affatto: `cs` era l'unico evento del portiere
    che la difficolta' della partita non toccava, quindi un portiere contro
    la capolista fuori casa e uno contro l'ultima in casa avevano la stessa
    probabilita' di porta inviolata. Ed e' il bonus piu' grande del ruolo.

    Non tocca `p_vote`: chi scende in campo non dipende da chi si affronta,
    dipende dalle scelte dell'allenatore, che sono gia' nella storia.

    **Quando ci sono le quote, `attack` qui non ha alcun effetto.** E' un
    fattore uguale per tutti i giocatori della stessa squadra nella stessa
    partita, e subito dopo `odds.allocate_attack` rinormalizza gli stessi tre
    eventi -- gf, rf, ass -- perche' la loro somma di squadra faccia
    `lambda_for`: qualunque costante comune si semplifica esattamente.
    Verificato, non dedotto: con e senza il fattore, le tre colonne
    coincidono a meno di 2e-16. Lo stesso vale per il fattore di squadra
    dentro `event_probabilities`.

    Non e' codice morto -- e' la strada di riserva, l'unica che resta quando
    le quote di quella partita mancano. Ma non va ritarata guardando il
    backtest con le quote: li' non muove niente, e ogni ora spesa a
    perfezionarla e' un'ora spesa su una moltiplicazione che si annulla.
    """
    if not (len(probabilities) == len(attack) == len(defense)):
        raise ValueError("probabilita', attacco e difesa devono avere la stessa lunghezza")

    out = probabilities.copy()
    out[list(TEAM_SCALED_EVENTS)] = out[list(TEAM_SCALED_EVENTS)].to_numpy() * attack[:, None]
    out["gs"] = out["gs"].to_numpy() * defense
    # il clip protegge l'elevamento a potenza: una base negativa con esponente
    # frazionario darebbe NaN, e un NaN qui si propaga fino al punteggio
    out["cs"] = np.clip(out["cs"].to_numpy(np.float64), 0.0, 1.0) ** defense
    return out
