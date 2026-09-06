"""Dalle quote dei bookmaker ai gol attesi delle due squadre.

Il mercato scommesse e' la miglior stima pubblica di come andra' una partita,
e finora il progetto ne usava una frazione minima: le tre quote 1X2 mediate
fra i bookmaker, ridotte a un solo numero (`p_win - p_lose`) e passate per una
regressione lineare sui gol. Tutto il resto era li' inutilizzato -- il mercato
over/under, che e' *esattamente* una domanda sul numero di gol, e l'handicap
asiatico, che e' *esattamente* una domanda su chi vince e di quanto.

Qui si fa il passo che mancava: si torna dalle probabilita' di mercato alla
coppia di gol attesi `(lambda_casa, lambda_trasferta)` che le genera. Da quella
coppia discendono in modo esatto le grandezze che al fantacalcio contano:

- i **gol subiti** di un portiere sono `lambda_avversario`, non una media
  storica corretta a occhio;
- la **porta inviolata** e' `P(l'avversario segna 0)`, che sotto Poisson vale
  `e^-lambda` -- una probabilita' calcolata, non un tasso storico shrinkato;
- la **forza offensiva** di una squadra e' `lambda_propria` rapportata alla
  media del campionato.

Sono i due eventi che decidono il ruolo dove il modello ha piu' margine da
recuperare, il portiere.

**Le quote di chiusura non sono una fuga di informazione.** Incorporano le
notizie sulle formazioni, ed e' corretto che lo facciano: la previsione di
FantaSquama si fa a ridosso del deadline, quando quelle notizie sono pubbliche
e l'utente le ha gia' lette. Usare le quote di apertura darebbe un modello
volutamente piu' ignorante dell'utente. Non «correggere» questo comportamento.

Nessuna rete: i CSV li scarica `fetch_fixtures.py`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from fantasquama.fixtures import _canonical, _season

# Fin dove si conta nella griglia dei risultati. Con lambda di Serie A (~1,3
# per squadra) la coda oltre gli 8 gol vale meno di 1e-6: allungarla non
# sposta nessuna cifra che poi si stampa, accorciarla si.
MAX_GOALS: int = 10

# Gol per squadra per partita in Serie A, il riferimento rispetto a cui una
# squadra e' «forte» o «debole» in attacco. Ricalcolato dai dati in
# `fit`, questo e' solo il ripiego per quando non c'e' abbastanza storia.
LEAGUE_GOALS: float = 1.30

# La correlazione di Dixon-Coles sui risultati bassi. La Poisson indipendente
# sbaglia sistematicamente su 0-0, 1-0, 0-1 e 1-1, che nel calcio sono una
# quota enorme dei risultati -- ed e' esattamente la zona che decide la porta
# inviolata. Il valore atteso in letteratura sta intorno a -0,13; qui viene
# stimato dai dati e questo e' solo il punto di partenza.
RHO_START: float = -0.13

# I mercati, in ordine di efficienza. Pinnacle lavora con margini bassi e
# limiti alti, ed e' il riferimento con cui gli altri si allineano; Bet365 e'
# il piu' capillare; la media di mercato e' l'ultima rete.
#
# L'ordine non e' teorico: nella stagione 2025-26 Pinnacle copre il 52% delle
# partite e Bet365 il 100%. Senza cascata, meta' stagione di verifica
# resterebbe senza gol attesi.
BOOKMAKERS: tuple[tuple[str, tuple[str, str, str], tuple[str, str]], ...] = (
    ("pinnacle_chiusura", ("PSCH", "PSCD", "PSCA"), ("PC>2.5", "PC<2.5")),
    ("pinnacle", ("PSH", "PSD", "PSA"), ("P>2.5", "P<2.5")),
    ("bet365_chiusura", ("B365CH", "B365CD", "B365CA"), ("B365C>2.5", "B365C<2.5")),
    ("bet365", ("B365H", "B365D", "B365A"), ("B365>2.5", "B365<2.5")),
    ("media", ("AvgH", "AvgD", "AvgA"), ("Avg>2.5", "Avg<2.5")),
)

# L'handicap asiatico serve solo a scegliere il punto di partenza
# dell'ottimizzazione, quindi basta la riga di chiusura di chi ce l'ha.
HANDICAP_COLUMNS: tuple[str, ...] = ("AHCh", "AHh")


def devig(quote: np.ndarray, method: str = "proporzionale") -> np.ndarray:
    """Quote decimali -> probabilita' a somma 1, tolto il margine del banco.

    Le quote grezze implicano probabilita' che sommano a piu' di 1: quella
    eccedenza e' il guadagno del bookmaker, e usarla senza toglierla gonfia
    ogni probabilita' della stessa quota.

    `proporzionale` divide per la somma: semplice, e giusto se il margine e'
    spalmato in modo uniforme. `shin` stima invece la quota di scommettitori
    informati e toglie il margine in modo non uniforme, il che conta quando
    c'e' un favorito netto -- li' il proporzionale sovrastima sistematicamente
    l'esito improbabile, e il portiere di una squadra sfavorita e' proprio uno
    dei casi in cui quella distorsione si vede.

    Lavora su una riga o su una matrice (una riga per partita).
    """
    quote = np.atleast_2d(np.asarray(quote, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        grezze = 1.0 / quote
    somma = grezze.sum(axis=1, keepdims=True)

    if method == "proporzionale":
        return np.squeeze(grezze / somma)
    if method != "shin":
        raise ValueError(f"metodo di de-vigging sconosciuto: {method!r}")

    # Shin: si cerca z, la quota di volume che viene da chi sa qualcosa, tale
    # che le probabilita' corrette tornino a sommare a 1. La formula chiusa
    # per n esiti sta in Shin (1993); qui si risolve numericamente perche' con
    # tre esiti costa niente ed e' molto piu' leggibile della forma chiusa.
    fuori = np.zeros_like(grezze)
    for i in range(len(grezze)):
        riga, totale = grezze[i], somma[i, 0]
        if not np.isfinite(totale) or totale <= 0:
            fuori[i] = np.nan
            continue

        def corrette(z: float) -> np.ndarray:
            radice = np.sqrt(np.maximum(z**2 + 4.0 * (1.0 - z) * riga**2 / totale, 0.0))
            return (radice - z) / (2.0 * (1.0 - z))

        if totale <= 1.0:  # nessun margine da togliere: niente da stimare
            fuori[i] = riga / totale
            continue
        z = optimize.brentq(lambda z: corrette(z).sum() - 1.0, 0.0, 0.9, xtol=1e-10)
        fuori[i] = corrette(z)
    return np.squeeze(fuori)


# Griglie e maschere precalcolate. L'ottimizzazione chiama `score_matrix`
# centinaia di volte per partita e migliaia di volte per stagione: rigenerare
# ogni volta un meshgrid e quattro confronti costa piu' del calcolo vero.
_CONTEGGI = np.arange(MAX_GOALS + 1)
_LOG_FATTORIALE = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, MAX_GOALS + 1)))])
_X, _Y = np.meshgrid(_CONTEGGI, _CONTEGGI, indexing="ij")
_CASA, _PARI, _FUORI = _X > _Y, _X == _Y, _X < _Y
_OVER = (_X + _Y) > 2
_ZERO_ZERO, _ZERO_UNO = (_X == 0) & (_Y == 0), (_X == 0) & (_Y == 1)
_UNO_ZERO, _UNO_UNO = (_X == 1) & (_Y == 0), (_X == 1) & (_Y == 1)


def score_matrix(lam: float, mu: float, rho: float = 0.0) -> np.ndarray:
    """P(x gol in casa, y gol fuori), come matrice (MAX_GOALS+1)^2.

    La correzione di Dixon-Coles sposta massa fra i quattro risultati bassi
    senza conservarla esattamente, quindi la matrice viene rinormalizzata:
    senza, le probabilita' che se ne ricavano non sommerebbero a 1 e l'errore
    finirebbe dritto nella porta inviolata, che di quella matrice e' una somma.
    """
    logaritmi_x = _CONTEGGI * np.log(max(lam, 1e-9)) - lam - _LOG_FATTORIALE
    logaritmi_y = _CONTEGGI * np.log(max(mu, 1e-9)) - mu - _LOG_FATTORIALE
    matrice = np.exp(logaritmi_x[:, None] + logaritmi_y[None, :])

    if rho:
        correzione = np.ones_like(matrice)
        correzione[_ZERO_ZERO] = 1.0 - lam * mu * rho
        correzione[_ZERO_UNO] = 1.0 + lam * rho
        correzione[_UNO_ZERO] = 1.0 + mu * rho
        correzione[_UNO_UNO] = 1.0 - rho
        # una correzione che rendesse negativa una probabilita' non e' una
        # correzione: e' un rho fuori dal dominio in cui il modello ha senso
        matrice = matrice * np.maximum(correzione, 0.0)

    totale = matrice.sum()
    return matrice / totale if totale > 0 else matrice


def market_probabilities(lam: float, mu: float, rho: float = 0.0) -> tuple[float, float, float, float]:
    """Da (lambda, mu) alle quattro probabilita' che il mercato quota.

    Ritorna `(vittoria casa, pareggio, vittoria fuori, over 2.5)`: sono le
    stesse quantita' di cui si hanno le quote, ed e' su queste che
    l'ottimizzazione confronta modello e mercato.
    """
    matrice = score_matrix(lam, mu, rho)
    return (
        float(matrice[_CASA].sum()),
        float(matrice[_PARI].sum()),
        float(matrice[_FUORI].sum()),
        float(matrice[_OVER].sum()),
    )


def _partenza(p_home: float, p_away: float, p_over: float, handicap: float | None) -> tuple[float, float]:
    """Un punto di partenza gia' vicino, cosi' l'ottimizzazione fa poco lavoro.

    Handicap asiatico e over/under sono i due mercati piu' efficienti, e
    rispondono direttamente alle due domande che servono: di quanto e'
    favorita una squadra, e quanti gol si aspettano in tutto. Dove
    l'handicap manca -- meta' della stagione 2025-26 -- la supremazia si
    ricava dalla differenza fra le due probabilita' di vittoria, che e' piu'
    grezza ma sempre disponibile.
    """
    totale = _totale_da_over(p_over)
    if handicap is not None and np.isfinite(handicap):
        supremazia = -float(handicap)
    else:
        supremazia = 2.0 * (p_home - p_away) if np.isfinite(p_home) and np.isfinite(p_away) else 0.0
    # meta' e meta', con un pavimento: una lambda nulla renderebbe il
    # logaritmo della verosimiglianza infinito
    return max((totale + supremazia) / 2.0, 0.05), max((totale - supremazia) / 2.0, 0.05)


def _totale_da_over(p_over: float) -> float:
    """Il totale di gol la cui Poisson da' quella probabilita' di over 2.5."""
    if not np.isfinite(p_over) or not 0.0 < p_over < 1.0:
        return 2.0 * LEAGUE_GOALS
    obiettivo = lambda t: (1.0 - np.exp(-t) * (1.0 + t + t**2 / 2.0)) - p_over
    return float(optimize.brentq(obiettivo, 0.05, 12.0, xtol=1e-8))


def solve_lambdas(
    p_home: float, p_draw: float, p_away: float, p_over: float,
    handicap: float | None = None, rho: float = 0.0,
) -> tuple[float, float]:
    """I gol attesi delle due squadre che riproducono le quote di mercato.

    Quattro vincoli (1, X, 2 e over 2.5) per due incognite: il sistema e'
    sovradeterminato e non ha soluzione esatta, quindi si minimizza lo scarto
    quadratico. E' un bene che sia sovradeterminato -- i quattro mercati si
    controllano a vicenda, e una quota storta pesa meno di quanto peserebbe
    se fosse l'unica.

    Se l'over/under manca, il vincolo sul totale sparisce e restano i tre
    dell'1X2: la soluzione e' piu' incerta sul numero di gol ma resta corretta
    su chi e' favorito.
    """
    partenza = _partenza(p_home, p_away, p_over, handicap)
    osservate = np.array([p_home, p_draw, p_away, p_over], dtype=np.float64)
    validi = np.isfinite(osservate)
    if validi.sum() < 2:
        return float("nan"), float("nan")

    def scarto(parametri: np.ndarray) -> float:
        lam, mu = np.exp(parametri)  # esponenziale: le lambda restano positive
        stimate = np.array(market_probabilities(lam, mu, rho))
        return float(((stimate[validi] - osservate[validi]) ** 2).sum())

    esito = optimize.minimize(
        scarto, np.log(partenza), method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 400},
    )
    lam, mu = np.exp(esito.x)
    return float(lam), float(mu)


@dataclass(frozen=True)
class Market:
    """I gol attesi di ogni partita, piu' le costanti con cui sono stati letti.

    `rho` e la media di campionato stanno qui e non fra le costanti del modulo
    perche' sono stimati dai dati, come `Difficulty` in `fixtures.py`: si
    ricalcolano a ogni esecuzione in pochi secondi, e tenerli in un file
    salvato darebbe una copia da sincronizzare a mano in cambio di niente.
    """

    lambdas: pd.DataFrame          # season, team, gameweek, lambda_for, lambda_against
    rho: float
    league_goals: float
    coverage: float                # quota di partite lette dalle quote
    source_counts: dict[str, int]  # da quale bookmaker, quante

    def clean_sheet(self, lambda_against: np.ndarray) -> np.ndarray:
        """P(l'avversario non segna), con la correzione sui risultati bassi.

        Non e' `e^-lambda` secco: la correzione di Dixon-Coles agisce
        soprattutto sullo 0-0, cioe' proprio dove vive la porta inviolata.
        """
        fuori = np.full(len(lambda_against), np.nan)
        for i, mu in enumerate(lambda_against):
            if not np.isfinite(mu):
                continue
            fuori[i] = float(score_matrix(self.league_goals, mu, self.rho)[:, 0].sum())
        return fuori


def _prima_disponibile(riga: pd.Series, gruppi: list[tuple[str, tuple[str, ...]]]) -> tuple[str, np.ndarray]:
    """Il primo gruppo di colonne completo, seguendo l'ordine di preferenza."""
    for nome, colonne in gruppi:
        if not set(colonne) <= set(riga.index):
            continue
        valori = pd.to_numeric(riga[list(colonne)], errors="coerce").to_numpy(np.float64)
        if np.isfinite(valori).all() and (valori > 1.0).all():
            return nome, valori
    return "", np.full(len(gruppi[0][1]), np.nan)


def read_odds(path: Path, method: str = "shin") -> pd.DataFrame:
    """Un CSV di football-data.co.uk -> probabilita' di mercato per partita.

    1X2 e over/under si scelgono per cascata e in modo indipendente: nella
    stagione 2025-26 meta' delle partite ha l'1X2 di Bet365 ma non quello di
    Pinnacle, e pretendere che i due mercati vengano dallo stesso bookmaker
    dimezzerebbe la copertura in cambio di una coerenza che l'ottimizzazione
    non richiede -- i quattro vincoli si mediano comunque fra loro.
    """
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["HomeTeam", "AwayTeam"])

    esiti = [(nome, colonne) for nome, colonne, _ in BOOKMAKERS]
    totali = [(nome, colonne) for nome, _, colonne in BOOKMAKERS]
    righe = []
    for _, riga in raw.iterrows():
        fonte, quote_1x2 = _prima_disponibile(riga, esiti)
        _, quote_ou = _prima_disponibile(riga, totali)

        p_home, p_draw, p_away = (
            devig(quote_1x2, method) if fonte else (np.nan, np.nan, np.nan)
        )
        # l'over/under e' un mercato a due esiti: stesso de-vigging, due colonne
        p_over = devig(quote_ou, method)[0] if np.isfinite(quote_ou).all() else np.nan

        handicap = np.nan
        for colonna in HANDICAP_COLUMNS:
            if colonna in riga.index and pd.notna(riga[colonna]):
                handicap = float(pd.to_numeric(riga[colonna], errors="coerce"))
                break

        righe.append({
            "home": _canonical(riga["HomeTeam"]), "away": _canonical(riga["AwayTeam"]),
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away, "p_over": p_over,
            "handicap": handicap, "source": fonte,
            "goals_home": pd.to_numeric(riga.get("FTHG"), errors="coerce"),
            "goals_away": pd.to_numeric(riga.get("FTAG"), errors="coerce"),
        })
    return pd.DataFrame(righe)


def estimate_rho(partite: pd.DataFrame) -> float:
    """La correlazione sui risultati bassi, per massima verosimiglianza.

    Si stima una volta su tutte le partite concluse, non partita per partita:
    e' una proprieta' di come si gioca a calcio, non di chi si affronta.

    Le lambda con cui si calcola la verosimiglianza vengono dalle quote lette
    con `rho = 0`. E' un passaggio circolare risolto con un giro solo: rho e'
    una correzione piccola, e ricavare le lambda con il rho stimato per poi
    ristimare rho sposta la terza cifra decimale. Il giro in piu' lo fa `fit`,
    che con il rho stimato ricalcola le lambda una seconda volta.
    """
    usabili = partite.dropna(subset=["lambda_home", "lambda_away", "goals_home", "goals_away"])
    if len(usabili) < 100:
        return 0.0

    lam = usabili["lambda_home"].to_numpy(np.float64)
    mu = usabili["lambda_away"].to_numpy(np.float64)
    x = usabili["goals_home"].to_numpy(int)
    y = usabili["goals_away"].to_numpy(int)

    def negativa(rho: float) -> float:
        totale = 0.0
        for i in range(len(lam)):
            if x[i] > MAX_GOALS or y[i] > MAX_GOALS:
                continue
            totale += np.log(max(score_matrix(lam[i], mu[i], rho)[x[i], y[i]], 1e-300))
        return -totale

    esito = optimize.minimize_scalar(negativa, bounds=(-0.35, 0.35), method="bounded",
                                     options={"xatol": 1e-4})
    return float(esito.x)


def fit(root: Path, fixtures: pd.DataFrame, method: str = "shin") -> Market:
    """Legge tutte le quote disponibili e ne ricava i gol attesi per squadra.

    Due passaggi sulle lambda, non uno: il primo le ricava con `rho = 0` per
    poter stimare rho, il secondo le rifa' con il rho stimato. Vedi
    `estimate_rho` per perche' un giro solo basta.

    `fixtures` serve solo a dire in che giornata cade ogni partita: i CSV
    delle quote portano la data e non la giornata, e dedurla dalle date e' il
    modo silenzioso di sbagliare che `load_fixtures` esiste per evitare -- un
    rinvio sfasa tutte le partite successive di quella squadra.

    Ritorna una riga per squadra per partita, la stessa forma di
    `load_fixtures`, cosi' l'aggancio all'archivio dei voti e' un merge su
    (stagione, squadra, giornata) e non un secondo abbinamento per nome.
    """
    partite = []
    for path in sorted(root.glob("odds_*.csv")):
        anno = int(path.stem.split("_")[1])
        lette = read_odds(path, method)
        if lette.empty:
            continue
        lette["season"] = _season(anno)
        partite.append(lette)

    if not partite:
        return Market(pd.DataFrame(), 0.0, LEAGUE_GOALS, 0.0, {})
    partite = pd.concat(partite, ignore_index=True)

    def risolvi(rho: float) -> None:
        coppie = [
            solve_lambdas(r.p_home, r.p_draw, r.p_away, r.p_over, r.handicap, rho)
            for r in partite.itertuples()
        ]
        partite["lambda_home"] = [c[0] for c in coppie]
        partite["lambda_away"] = [c[1] for c in coppie]

    risolvi(0.0)
    rho = estimate_rho(partite)
    risolvi(rho)

    lette = partite["lambda_home"].notna()
    media = float(
        pd.concat([partite.loc[lette, "lambda_home"], partite.loc[lette, "lambda_away"]]).mean()
    ) if lette.any() else LEAGUE_GOALS

    # una riga per squadra: quella di casa subisce le lambda di chi gioca fuori
    # e viceversa. E' la forma che serve al portiere, che dei gol attesi guarda
    # quelli dell'avversario.
    casa = partite.assign(
        team=partite["home"], opponent=partite["away"], home=True,
        lambda_for=partite["lambda_home"], lambda_against=partite["lambda_away"],
    )
    fuori = partite.assign(
        team=partite["away"], opponent=partite["home"], home=False,
        lambda_for=partite["lambda_away"], lambda_against=partite["lambda_home"],
    )
    colonne = ["season", "team", "opponent", "home", "lambda_for", "lambda_against", "source"]
    lambdas = pd.concat([casa[colonne], fuori[colonne]], ignore_index=True)

    # La giornata dal calendario, che e' l'unica fonte che la dichiara. Il
    # campo fa parte della chiave: due squadre si affrontano due volte a
    # stagione, quindi (stagione, squadra, avversario) da solo pesca sia
    # l'andata sia il ritorno e raddoppia le righe in silenzio.
    lambdas = lambdas.merge(
        fixtures[["season", "team", "opponent", "home", "gameweek"]],
        on=["season", "team", "opponent", "home"], how="left", validate="one_to_one",
    )

    return Market(
        lambdas=lambdas,
        rho=rho,
        league_goals=media,
        coverage=float(lambdas["lambda_for"].notna().mean()),
        source_counts=partite["source"].value_counts().to_dict(),
    )


def align(market: Market, archive: pd.DataFrame) -> pd.DataFrame:
    """Le lambda della partita di ogni riga dell'archivio, allineate per posizione.

    Dove le quote mancano le due colonne restano NaN, ed e' il segnale con cui
    gli stadi a valle sanno di dover ricadere sulla stima empirica.
    """
    if market.lambdas.empty:
        return pd.DataFrame(
            {"lambda_for": np.nan, "lambda_against": np.nan}, index=archive.index
        )
    lam = market.lambdas.dropna(subset=["gameweek"]).copy()
    lam["gameweek"] = lam["gameweek"].astype(int)
    # `validate` non e' decorativo: una chiave non unica qui moltiplicherebbe
    # le righe dell'archivio, e un frame piu' lungo di quello che descrive
    # sfasa ogni allineamento posizionale a valle
    merged = archive[["season", "team", "gameweek"]].merge(
        lam[["season", "team", "gameweek", "lambda_for", "lambda_against"]],
        on=["season", "team", "gameweek"], how="left", validate="many_to_one",
    )
    # il merge azzera l'indice: si rimette per posizione, come ovunque
    merged.index = archive.index
    return merged[["lambda_for", "lambda_against"]]


def apply_goalkeeper(
    probabilities: pd.DataFrame, roles: pd.Series,
    lambda_against: np.ndarray, market: Market,
) -> pd.DataFrame:
    """Sostituisce gol subiti e porta inviolata con quello che dice il mercato.

    Sono le due voci che il mercato prezza quasi esattamente, e le due che
    decidono il fantavoto di un portiere. La stima che sostituiscono era il
    tasso storico del portiere shrinkato verso la media di ruolo: misurata
    sulla stagione 2025-26 dava un Brier di 0,2252 sulla porta inviolata,
    **peggio della costante che predice sempre la frequenza media** (0,2195).
    Quella dedotta dalle quote da' 0,2048 ed e' calibrata entro pochi punti
    in ogni fascia.

    Non e' una perdita di personalizzazione: i gol subiti sono una proprieta'
    della squadra e dell'avversario, non del portiere -- e quel poco che il
    portiere ci mette il mercato lo ha gia' prezzato nella forza della
    squadra. Il portiere resta se stesso in tutto il resto: voto, rigori
    parati, e soprattutto la probabilita' di giocare.

    Tocca solo i portieri e solo dove le quote ci sono: le altre righe
    restano quelle di prima.
    """
    if len(lambda_against) != len(probabilities) or len(roles) != len(probabilities):
        raise ValueError("lambda e ruoli devono avere una riga per riga di probabilities")

    portieri = roles.fillna("").astype(str).to_numpy() == "P"
    quotate = np.isfinite(lambda_against)
    tocca = portieri & quotate
    if not tocca.any():
        return probabilities

    out = probabilities.copy()
    gs = out["gs"].to_numpy(np.float64).copy()
    cs = out["cs"].to_numpy(np.float64).copy()
    gs[tocca] = lambda_against[tocca]
    cs[tocca] = market.clean_sheet(lambda_against[tocca])
    out["gs"], out["cs"] = gs, cs
    return out


# Quanto l'allocazione puo' spostare una squadra rispetto a quello che la
# somma dei suoi giocatori direbbe. Non e' un tetto sulla forza della squadra
# -- quella la fissa `lambda_for` -- ma una protezione contro il denominatore:
# a inizio stagione la somma attesa di una squadra puo' essere quasi zero, e
# senza estremi il rapporto esploderebbe.
ALLOCATION_MIN: float = 0.25
ALLOCATION_MAX: float = 4.0

# Gli eventi che si spartiscono i gol della squadra. Gli assist seguono gli
# stessi gol -- non si puo' servire un assist per una rete che non arriva --
# quindi si riscalano con lo stesso fattore invece di avere un vincolo loro.
ALLOCATED_EVENTS: tuple[str, ...] = ("gf", "rf", "ass")


def allocate_attack(
    probabilities: pd.DataFrame, archive: pd.DataFrame, lambda_for: np.ndarray
) -> pd.DataFrame:
    """Spartisce i gol attesi della squadra fra i suoi giocatori.

    Prima ogni giocatore veniva scalato per la forza della squadra, e per
    due volte: una col ritmo storico dentro `event_probabilities`, una col
    fattore di mercato in `apply_match_context`. Due difetti in uno.

    Il primo e' il doppio conteggio. Il secondo e' che la moltiplicazione non
    conserva niente: scalando tutti per 1,8 la squadra segna 1,8 volte i gol
    che segnera' davvero, e i tetti a 2x nascondevano il sintomo senza
    togliere la causa.

    Qui il vincolo si impone per costruzione. La somma dei gol attesi dei
    giocatori che ci si aspetta in campo -- ciascuno pesato per la sua
    probabilita' di scendere -- deve valere `lambda_for`, che e' quanto il
    mercato dice che quella squadra segnera' in quella partita. Ogni giocatore
    tiene la propria QUOTA e il totale viene dalle quote.

    Ne discende gratis il comportamento che prima non c'era: se manca il
    bomber, la sua quota si ridistribuisce sugli altri invece di sparire.
    La squadra segnera' lo stesso quei gol, li fara' solo qualcun altro.

    Tocca solo le squadre-giornata per cui esiste una lambda: le altre
    restano come sono.
    """
    if len(lambda_for) != len(probabilities) or len(archive) != len(probabilities):
        raise ValueError("lambda e archivio devono avere una riga per riga di probabilities")

    out = probabilities.copy()
    p_vote = out["p_vote"].to_numpy(np.float64)
    # I gol che la squadra si aspetta dai suoi giocatori: il tasso per
    # presenza moltiplicato per la probabilita' di essere in campo. E' la
    # stessa grandezza di `lambda_for`, quindi il rapporto fra le due e'
    # adimensionale ed e' esattamente il fattore che serve.
    attesi = (out["gf"].to_numpy(np.float64) + out["rf"].to_numpy(np.float64)) * p_vote

    gruppi = pd.DataFrame({
        "season": archive["season"].astype(str).to_numpy(),
        "team": archive["team"].astype(str).to_numpy(),
        "gameweek": pd.to_numeric(archive["gameweek"], errors="coerce").to_numpy(),
        "attesi": attesi,
        "lambda_for": lambda_for,
    })
    somme = gruppi.groupby(["season", "team", "gameweek"], observed=True)["attesi"].transform("sum")

    fattore = np.where(
        np.isfinite(lambda_for) & (somme.to_numpy() > 1e-9),
        lambda_for / somme.to_numpy().clip(1e-9),
        1.0,
    )
    fattore = np.clip(fattore, ALLOCATION_MIN, ALLOCATION_MAX)

    out[list(ALLOCATED_EVENTS)] = out[list(ALLOCATED_EVENTS)].to_numpy() * fattore[:, None]
    return out
