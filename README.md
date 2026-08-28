# fantasquama-data

I dati che l'app [FantaSquama](https://github.com/RiccardoRomano9/FantaSquama) legge.

`serieA.json` si aggiorna da solo: un lavoro schedulato scarica le probabili
formazioni e le applica al file base. Un secondo lavoro controlla due volte al
giorno i voti Excel di Fanta.Soccer: quando trova una giornata nuova, la
archivia in un repository privato e rigenera il bundle per la giornata
successiva. L'indirizzo che l'app interroga è

```
https://raw.githubusercontent.com/RiccardoRomano9/fantasquama-data/main/serieA.json
```

## Cosa c'è dentro

Per ogni giocatore: ruolo, squadra, avversario, la probabilità di ogni evento
(gol, assist, ammonizione…) e la probabilità di scendere in campo. **Non ci
sono i fantapunti**: dipendono dal regolamento della lega, e li calcola il
telefono. È la stessa separazione che rende configurabili le regole.

Non ci sono nemmeno gli ultimi fantavoti presi: quelli vengono dall'archivio
storico, che è ad uso personale di chi lo scarica, e restano nel file dentro
all'app.

## Come si rigenera

Questa cartella non si scrive a mano: la produce `publish/build.py` dal repo
principale. Modificare un file qui significa vederselo sovrascritto al
prossimo aggiornamento del modello.

## Fonti

- Probabili formazioni: [sosfanta.com](https://www.sosfanta.com)
- Quote 1X2, player props e mercati partita pre-match/live:
  [The Odds API](https://the-odds-api.com)
- Calendario e stemmi: [football-data.org](https://www.football-data.org)
- Ruoli e quotazioni: listone ufficiale di Fantacalcio.it
- Identità e date di nascita: Wikipedia in italiano e Wikidata
- URL dei ritratti: CDN di Gazzetta (nel repository restano solo gli URL)

## Foto dei giocatori

`fetch_player_photos.py` tratta separatamente identità e immagine:

1. prova i titoli Wikipedia esatti in batch da 20;
2. recupera entità Wikidata in batch da 40 e accetta solo esseri umani con
   occupazione/descrizione da calciatore e data di nascita con precisione al
   giorno;
3. usa la ricerca Wikipedia, una persona alla volta, soltanto per i titoli
   esatti non risolti; se non esiste una pagina italiana, prova la ricerca
   entità Wikidata;
4. genera le varianti conservative del nome e verifica con `GET` ogni URL
   Gazzetta prima di pubblicarlo.

Se il nome anagrafico completo non ha una pagina esatta, viene provato in un
secondo batch un titolo più corto ancorato al nome del listone (per esempio
`Jurgen Peter Ekkelenkamp` → `Jurgen Ekkelenkamp`). Non vengono mai create
combinazioni arbitrarie nome/cognome: il candidato deve restare univoco e
superare gli stessi controlli Wikidata.

Per i cognomi composti prova prima lo slug con il trattino conservato (per
esempio `loftus-cheek`) e poi la variante storica con underscore. Questo evita
falsi 404 senza rendere meno rigorosa la validazione dell'immagine.

Non viene fatta alcuna query SPARQL e non serve una API key. Se due candidati
hanno la stessa confidenza, oppure il confronto del nome non supera la soglia,
il giocatore resta senza URL. L'app mostra in quel caso il fallback locale.

Esecuzione locale equivalente al workflow:

```bash
python -m unittest discover -s tests -p 'test_fetch_player_photos.py' -v
python fetch_player_photos.py \
  --base serieA-base.json \
  --derived serieA.json \
  --cache player-photo-cache.json \
  --overrides player-photo-overrides.json
```

Il file derivato viene sincronizzato soltanto nei campi `fullName`, `photoURL`
e `photoProviderID`: titolarità, notizie e altri dati live già presenti non
vengono alterati. Se almeno uno di questi campi cambia, viene avanzato anche
`generatedAt`, così un'app che ha già il JSON in cache adotta l'aggiornamento.

### Cache

`player-photo-cache.json` è versionato e indicizzato per l'`id` stabile del
listone. Una voce tipica è:

```json
{
  "version": 1,
  "players": {
    "1234": {
      "inputHash": "92cfe58c3a9cd70ac985",
      "status": "valid",
      "checkedAt": "2026-08-26T10:00:00Z",
      "fullName": "Nikola Krstović",
      "birthDate": "2000-04-05",
      "wikidataID": "Q123456",
      "wikipediaTitle": "Nikola Krstović",
      "photoURL": "https://images2.gazzettaobjects.it/assets-mc/calcio/giocatori/nikola_krstovic_05042000.png",
      "photoProviderID": "gazzetta:nikola_krstovic_05042000",
      "image": {
        "sha256": "...",
        "width": 370,
        "height": 444,
        "bytes": 85431,
        "contentType": "image/png"
      },
      "attempts": []
    }
  }
}
```

Gli stati negativi (`not_found`, `ambiguous`, `gazzetta_404`,
`image_invalid`) vengono riprovati dopo 30 giorni; i positivi dopo 90. Un
cambio di nome, squadra, ruolo o override invalida subito la singola voce.
Gli errori transitori non sostituiscono mai una voce valida.

### Override manuali

`player-photo-overrides.json` è anch'esso versionato. Le chiavi sono sempre
gli ID del listone, mai i nomi. Gli ID di rose passate possono restare nel
file: vengono validati ma ignorati finché non sono nella rosa corrente. Sono
supportati questi casi:

```json
{
  "version": 1,
  "players": {
    "1234": {
      "wikidataID": "Q123456",
      "gazzettaName": "Nikola Krstovic",
      "note": "QID verificato manualmente"
    },
    "2345": {
      "fullName": "Nome Cognome",
      "birthDate": "2001-12-31"
    },
    "3456": {
      "photoURL": "https://images2.gazzettaobjects.it/.../file.png",
      "photoProviderID": "gazzetta:file"
    },
    "4567": {
      "skip": true,
      "note": "due omonimi non separabili"
    }
  },
  "invalidImageSHA256": [
    "hash_sha256_di_un_placeholder_confermato"
  ]
}
```

Anche un `photoURL` manuale viene scaricato e validato: l'override decide
l'identità, non aggira i controlli del contenuto.

### Rate limit e validazione

Il client invia uno User-Agent identificabile, attende almeno 250 ms fra le
richieste Wikimedia e 350 ms fra quelle al CDN, usa timeout di 25 secondi e
fino a tre retry con backoff esponenziale, jitter e rispetto di `Retry-After`.
Le chiamate MediaWiki portano anche `maxlag=15`, così il bot cede il passo
quando le repliche sono troppo indietro. Dopo tre fallimenti transitori
consecutivi apre un circuit breaker per quella fonte e completa il report
senza fallire l'intero job.

Un'immagine è valida soltanto se la risposta è `image/png` (o `image/x-png`),
resta sotto 2 MB, porta firma PNG, chunk `IHDR`, dimensioni fra 80 e 4000 px e
chunk finale `IEND`. Sono inoltre rifiutati gli hash nella denylist e lo stesso
identico file servito da URL di giocatori diversi, perché è un forte segnale di
placeholder. Una pagina HTML con status 200 non supera questi controlli.

Il report completo finisce nel log e nel Job Summary di GitHub Actions: totale
valido/fallback, ambigui, non trovati, immagini non valide, errori transitori e
tutti gli URL che hanno risposto 404.

## Impostarlo, una volta sola

Con la CLI di GitHub (`gh`), dalla radice del repo principale:

```bash
# 1. genera la cartella del repo pubblico
backtest/.venv/bin/python publish/build.py --out ../fantasquama-data

# 2. crea il repo e caricalo
cd ../fantasquama-data
git init -b main && git add -A && git commit -m "Primo file base"
gh repo create fantasquama-data --public --source=. --push

# 3. permetti al lavoro schedulato di committare
gh api -X PUT repos/RiccardoRomano9/fantasquama-data/actions/permissions/workflow \
  -f default_workflow_permissions=write

# 4. provalo subito, senza aspettare il cron
gh workflow run aggiorna
gh run watch
```

## Quote e piano gratuito

Il lavoro schedulato separa tre ritmi diversi:

- le probabili formazioni si aggiornano spesso, soprattutto nei giorni di gara;
- le quote 1X2 si riscaricano al massimo una volta ogni 24 ore;
- player props e mercati avanzati si provano solo in due finestre per giornata:
  tra quattro e due giorni dal primo calcio d'inizio, e poi nelle ultime trenta
  ore prima della prima partita.

Questa scansione fa trovare al modello i segnali bookmaker già pronti quando si
studia la formazione, ma impedisce ai cron ravvicinati del weekend di consumare
il piano gratuito. I props usati sono `player_goal_scorer_anytime`,
`player_assists`, `player_to_receive_card` e `btts`; vengono salvati dentro
`serieA.json` e riusati finché la giornata resta la stessa.

Il budget ha anche un tetto interno: i props si fermano a 420 crediti stimati
nel mese, lasciando margine ai 31 crediti circa delle quote 1X2 giornaliere
rispetto ai 500 crediti mensili del piano free.

Il passo 3 è l'unico che si dimentica: senza, il lavoro gira, produce il file
e poi fallisce al `git push`.

Per aggiornare anche le probabilità ricavate dalle quote, aggiungi la chiave di
The Odds API come secret del repo pubblico:

```bash
gh secret set THE_ODDS_API_KEY --repo RiccardoRomano9/fantasquama-data
```

Il lavoro riusa le quote già pubblicate per la stessa giornata e richiama
The Odds API al massimo una volta ogni 24 ore. Il job può quindi continuare a
girare spesso per le probabili senza consumare crediti odds a ogni passaggio.

## Spiegazione Coach Squama

La scheda Consigli puo' ricevere una nota generata da DeepSeek in stile articolo
di redazione. Il testo usa markdown controllato: titoletti, paragrafi brevi e
nomi dei giocatori in maiuscolo/grassetto. Anche questa viene prodotta dal job
pubblico, non dall'app: il telefono legge solo il testo gia' salvato in
`serieA.json` e lo renderizza senza mostrare gli asterischi.

La chiamata parte una volta sola per giornata, solo nel giorno della prima
partita, dopo le 10:00 italiane e prima del calcio d'inizio. Se il campo
`tipsExplanation` esiste gia' per quella stagione/giornata, il job non richiama
DeepSeek e riusa il testo. Se manca la secret, salta la generazione.

Le top 3 restano ordinate per rendimento atteso nel ruolo. Gli 11 del Coach
Squama, invece, sono una selezione diversa: cercano giocatori che possono
sorprendere per matchup, probabilita' di voto, piazzati e bonus possibili.

Per abilitarla:

```bash
gh secret set DEEPSEEK_API_KEY --repo RiccardoRomano9/fantasquama-data
```

Da lì in poi non serve toccarlo per le probabili. Per automatizzare anche i
voti bisogna configurare una sola volta l'archivio privato e la sua chiave di
deploy: le istruzioni sono nella guida del repository principale, perché la
chiave privata non deve mai comparire in questo repository pubblico.
