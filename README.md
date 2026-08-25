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
- Calendario e stemmi: [football-data.org](https://www.football-data.org)
- Ruoli e quotazioni: listone ufficiale di Fantacalcio.it

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

Il passo 3 è l'unico che si dimentica: senza, il lavoro gira, produce il file
e poi fallisce al `git push`.

Da lì in poi non serve toccarlo per le probabili. Per automatizzare anche i
voti bisogna configurare una sola volta l'archivio privato e la sua chiave di
deploy: le istruzioni sono nella guida del repository principale, perché la
chiave privata non deve mai comparire in questo repository pubblico.
