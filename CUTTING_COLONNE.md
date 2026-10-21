# Processo di taglio delle colonne — BRFSS 2015 (`_MICHD`)

Documento di passaggio per il gruppo: spiega **da dove siamo partiti**, **come abbiamo trattato i
valori mancanti**, **perché abbiamo tagliato le colonne** e **dove siamo arrivati**.
Tutto quello che scrivete qui è replicabile dal notebook `notebooks/Giovanni.ipynb`.

> **Vincoli del progetto**: solo Python standard + NumPy (no pandas, no sklearn).
> Regola aurea: **ogni statistica (mediana, moda, media, deviazione standard, categorie one-hot)
> va calcolata solo sul train** e applicata identica al test.

---

## 0. TL;DR

| passo | colonne | NaN totali | cosa è successo |
|---|---|---|---|
| dataset originale | **321** (+`Id`) | 44,8% | BRFSS 2015 grezzo |
| 1. drop "a priori" | **273** | 45,6% | 48 colonne eliminate (identificativi, pesi, fisso/cellulare, leakage, grezze ridondanti) |
| 2. pulizia dei valori | 273 | 45,6% | zeri strutturali, codici "nessuno"→0, codici "non so/rifiuta"→NaN |
| 3. taglio 60% NaN | **153** | 8,3% | 123 colonne con >60% NaN eliminate, meno 3 candidate; `INSULIN`→`INSULIN_CAT` |
| 4. taglio ridondanti | **129** | 8,3% | 24 copie determinate buttate (nessuna perdita di informazione) |

(I NaN sono 44,8% sul dataset grezzo e 45,6% dopo il drop delle 48 colonne "a priori": il %
sale leggermente perché le colonne eliminate erano quelle *più* complete.)

Restano due matrici pronte per il modello:

- `x_base` → **127 colonne** (senza `CVDASPRN` e `DRADVISE`): è il **modello di riferimento**
- `x_train` → **129 colonne** (con le due candidate): serve per l'*ablation*

**Prossimo passo**: preprocessing (imputazione + indicatori di missing + one-hot + standardizzazione)
e poi cross-validation a 5 fold stratificata. Le 3 decisioni ancora aperte sono elencate in §8.

---

## 1. Il dataset

- **BRFSS 2015**: questionario sanitario telefonico USA (Behavioral Risk Factor Surveillance System).
- Train: **328.135** persone × **321** feature (+ colonna `Id`); Test: **109.379** righe.
- Target **`_MICHD`** ∈ {-1, +1}: "infarto / malattia coronarica". **8,83% positivi** (28.975 persone)
  → classi molto sbilanciate (~1:10). Qualsiasi metrica va letta con questa lente.
- Il questionario è **condizionale**: molte domande sono poste solo a chi ha risposto "sì" a una
  domanda precedente, e i moduli opzionali sono scelti dai singoli Stati.
- **44,8% di tutte le celle sono NaN** (147 colonne con più del 50% di NaN, 99 con più del 90%).
  Train e test hanno la *stessa* struttura di NaN (differenza max 0,4%).

---

## 2. I NaN non sono casuali: le 4 cause

| # | causa | esempio | colonne | trattamento |
|---|---|---|---|---|
| 1 | **Versione fisso vs cellulare** del questionario | `CTELENUM` vs `CTELNUM1` | ~14 | **eliminate**: il NaN dice solo "sono stato chiamato al cellulare", non parla di salute |
| 2 | **Intervista interrotta** (`DISPCODE`=1200, ~15% delle righe) | `INCOME2`, `EXERANY2`, `_BMI5` | ~40 | NaN veri → **imputazione** + indicatore di missing |
| 3 | **Domande condizionali** (solo se la risposta precedente è "sì") | `AVEDRNK2` solo a chi beve | ~45 | **0** dove è una quantità, altrimenti categoria "non applicabile" |
| 4 | **Moduli opzionali** scelti dagli Stati (asma, vista, memoria, depressione, contesto sociale, screening) | `CRGVHRS1`, `ASRCHKUP` | ~100 | **soglia sui NaN** + lista di "candidate" da valutare |

Due scoperte importanti fatte in fase di EDA:

- **Il NaN è spesso informativo**: il NaN di `DIABAGE2` significa "non diabetico"
  (tasso di infarto **6,9%** contro **22%** fra chi ha risposto). Quindi "mancante" ≠ "assente":
  il modello deve poter distinguere "0 vero" da "non so".
- **Le figlie ripetono le madri**: `ASTHNOW` ha NaN solo per chi non ha l'asma (lo dice già `ASTHMA3`);
  `PREGNANT` ha NaN solo per gli uomini (lo dice `SEX`); `_FLSHOT6` e `_PNEUMO2` hanno NaN solo
  sotto i 65 anni (lo dice `_AGE80`).

---

## 3. Fase 1 — Pulizia dei valori (prima di tagliare le colonne)

Da sola, questa fase **non toglie nessuna colonna**, ma senza di lei il taglio sarebbe sbagliato:
con i codici speciali ancora dentro, "quasi nessuno risponde" sarebbe vero per la metà sbagliata
delle colonne. Le tre funzioni sono in `cleaning.py` e si applicano **nell'ordine** (i passi 1-2
leggono i codici originali delle domande madri, il passo 3 li cancella).

### 3.1 Zeri strutturali — `fill_structural_zeros`

Se una domanda *figlia* (una quantità) non è stata posta perché la risposta alla *madre* è "no",
la quantità vera è **0**. Si mette 0 **solo dove la madre dice esplicitamente "no"**: se la madre è
NaN (intervista interrotta) la figlia resta NaN, perché lì il dato manca davvero.

| figlie | madre | 0 se la madre vale |
|---|---|---|
| `AVEDRNK2`, `DRNK3GE5`, `MAXDRNKS` | `ALCDAY5` | 888 = nessun drink negli ultimi 30 gg |
| `PA*`, `_MINAC*`, `METVL*`, `ACTIN*` (attività 1) | `EXERANY2` | 2 = non fa attività |
| `PA*`, `_MINAC*`, `METVL*`, `ACTIN*` (attività 2) | `EXERANY2` / `EXRACT21` | 2 / 88 = nessuna seconda attività |
| `POORHLTH` | `PHYSHLTH` e `MENTHLTH` | entrambe 88 = nessun giorno |

**Caso speciale `DIABAGE2`**: "età alla diagnosi del diabete" viene trasformata in
**anni con il diabete** = `_AGE80` − `DIABAGE2`, con **0 per i non diabetici**
(`DIABETE3` ∈ {2,3,4}). Mettere 0 nell'età alla diagnosi non avrebbe nessun senso.
Fra i diabetici il tasso di infarto sale dal 17% (≤4 anni di diabete) al 29% (>18 anni).

### 3.2 Codici "nessuno / mai" → 0 — `recode_none_to_zero`

Lasciati così, un modello lineare leggerebbe "88 giorni di malessere" in un mese di 30.

- `88` → 0: `PHYSHLTH`, `MENTHLTH`, `POORHLTH` (giorni), `CHILDREN` (figli), `DRNK3GE5` (binge)
- `888` → 0: `ALCDAY5` (giorni con alcol), `STRENGTH` (allenamenti)
- `555` → 0: `FRUITJU1`, `FRUIT1`, `FVBEANS`, `FVGREEN`, `FVORANG`, `VEGETAB1`

### 3.3 Codici "non so / rifiuta" → NaN — `find_special_codes` + `special_codes_to_nan`

Il BRFSS codifica le non-risposte **con numeri**: 7/77/777/7777 = "non so", 9/99/999/9999 = "rifiuta".
La lunghezza del codice dipende da quante cifre ha la risposta.

**Regola applicata** (calcolata sul train, applicata identica al test):

- se il **massimo** della colonna è 9, 99, 999, … → i valori 9…9 e 7…7 diventano NaN;
- con una sola cifra il 7 è un codice **solo se il 6 non compare** (in `EMPLOY1` il 7 è
  "pensionato", una risposta valida: la regola lo lascia stare);
- le colonne continue calcolate dal CDC (BMI, minuti di attività, somme di frutta/verdura, pesi)
  sono **esentate**: non hanno codici e non vanno toccate (lista `NO_CODES`);
- alcuni codici fuori schema sono indicati a mano (`_AGEG5YR`=14, `_AGE65YR`=3, `DROCDY3_`=900,
  `_DRNKWEK`=99900, `ASERVIST`/`ASDRVIST`=98).

---

## 4. Fase 2 — Il taglio delle colonne

### 4.1 Colonne eliminate "a priori" (48) — le 321 → 273

| gruppo | n | esempi | perché |
|---|---|---|---|
| identificativi e date | 8 | `Id`, `_STATE`, `FMONTH`, `IDATE`, `IDAY`, `IYEAR` | non dicono nulla sulla persona |
| gestione della chiamata | 11 | `DISPCODE`, `SEQNO`, `_PSU`, `CTELENUM`, `QSTVER`, `QSTLANG` | descrivono il campionamento, non la salute |
| pesi statistici del sondaggio | 7 | `_STSTR`, `_STRWT`, `_RAWRAKE`, `_WT2RAKE`, `_DUAL*`, `_LLCPWT` | servono solo per stimare la popolazione USA |
| versione fisso/cellulare | 14 | `COLGHOUS`, `LADULT`, `NUMPHON2`, `CTELNUM1`, `CELLFON2`, `CSTATE`, `LANDLINE` | il NaN dice solo il tipo di telefono |
| **leakage** | 1 | **`HAREHAB1`** | compilata **solo** da chi ha già avuto un infarto (target=1 nel 100% dei 773 casi) → la predice perfettamente, ma è un'informazione *post-outcome* |
| grezze ridondanti | 8 | `WEIGHT2`, `HEIGHT3`, `EXEROFT1/2`, `EXERHMM1/2`, `FLSHTMY2`, `HIVTSTD3` | esiste già la versione pulita del CDC (`WTKG3`, `HTM4`, `_BMI5`, `PAFREQ*_`, `PADUR*_`) |

### 4.2 Taglio automatico: le colonne con > 60% di NaN (123) — le 273 → 153

`identify_columns_to_drop(x_train, cols, 0.6)` → **123 colonne**. Tre ragioni per buttarle:

1. **Quasi nessuno risponde.** `CRGVHRS1` (chi si prende cura di un anziano) ha il **94,9%** di NaN:
   hanno un valore solo il 5,1% × 328.135 ≈ **16.700 persone**. Anche se fosse utilissima,
   potrebbe cambiare la previsione per 5 persone su 100; per le altre 95 vale "non so".
2. **Il NaN non parla di salute.** I moduli sono opzionali e ogni Stato sceglie: il NaN significa
   "il tuo Stato non l'ha chiesto" → è un segnale **geografico**, non clinico.
3. **Le figlie ripetono la madre** (vedi §2): `ASTHNOW`, `HAVARTH3`/`_DRDXAR1`, `DIABETE3`/`DIABAGE2`,
   `PREGNANT`, `_FLSHOT6`, `_PNEUMO2` …

**Un conto per capire il peso di una colonna** (idea grezza, non un criterio di selezione):
`spread × quota di risposta`, dove `spread` = quanto cambia il tasso di infarto fra i valori.

| colonna | spread | rispondono | peso |
|---|---|---|---|
| `GENHLTH` (tenuta) | 5,37 | 99,7% | **5,36** |
| `DRADVISE` | 6,03 | 9,3% | 0,56 |
| `CVDASPRN` | 6,86 | 4,3% | 0,29 |
| `INSULIN` | 3,98 | 6,6% | 0,26 |
| `CRGVHRS1` | 1,09 | 5,1% | 0,06 |
| `ASYMPTOM` | 6,95 | 0,14% | 0,01 |

`ASYMPTOM` sembra forte in classifica ma riguarda **77 infarti** su 328.135 persone: la decisione
vera la prende la cross-validation, non questo numero.

> ⚠️ **La soglia del 60% è fragile**: `BPMEDS` è al **59,9%** di NaN, cioè *sulla soglia* (tenuta
> per 0,1 punti percentuali!). Da confrontare in CV con soglie 0,55 / 0,65.

### 4.3 Le 3 colonne "contese": `INSULIN`, `CVDASPRN`, `DRADVISE`

Non buttate a mano: **fuori dal dataset base, dentro l'*ablation***.

**`INSULIN`** — l'unica con una ragione clinica specifica: fra i diabetici **chi usa insulina ha
il 28,0% di infarti contro il 18,96%** di chi non la usa. Ma il suo NaN **mescola due cose diverse**
("non sono diabetico" e "sono diabetico ma il mio Stato non ha somministrato il modulo"), quindi
da sola è ambigua: serve `DIABETE3` per separare i due casi.

**`CVDASPRN` e `DRADVISE`** (aspirina) hanno gli spread più alti del gruppo, ma coprono solo il
4% e il 9% della popolazione. Erano sospettate di *leakage* (l'aspirina viene prescritta **dopo**
un infarto → più che una causa, una conseguenza). Il controllo dice che il sospetto è **solo
parzialmente fondato**:

- **non è leakage strutturale**: la domanda non è posta solo a chi ha avuto l'infarto.
  Quota di risposta quasi identica nei due gruppi (`CVDASPRN`: 4,53% fra i positivi, 4,25% fra i
  negativi; `DRADVISE`: 11,4% e 9,1%). Chi ha fatto il modulo ha risposto, indipendentemente dal target;
- **è però un proxy post-outcome fortissimo**: chi prende aspirina ha **4,6×** la prevalenza di
  infarto di chi non la prende (20,84% vs 4,48%); per `DRADVISE` 20,72% vs 6,50%.

→ Decisione: si misura in CV. Se la CV dice che guadagnano, le si aggiunge al modello finale e lo
si dichiara apertamente nel report; se non guadagnano, avremo la prova che il sospetto era giusto.
Toglierle a mano senza misurare sarebbe un'affermazione non supportata dai dati.

### 4.4 `INSULIN` ricostruito in 5 categorie (`INSULIN_CAT`)

| valore | significato | n | tasso infarti |
|---|---|---|---|
| 0 | non diabetico (`DIABETE3` ∈ {2,3,4}) | 285.433 | 6,86% |
| 1 | diabetico, ma il modulo non è stato somministrato | 20.535 | 22,35% |
| 2 | ignoto (`DIABETE3` mancante) | 479 | 12,32% |
| 3 | diabetico **+ insulina** | 7.133 | **27,98%** |
| 4 | diabetico, **senza insulina** | 14.555 | 18,96% |

Le regole sono basate **solo sui valori** (non su statistiche aggregate) → applicate **identiche**
a train e test. `INSULIN` (la versione grezza) viene buttata, `INSULIN_CAT` la sostituisce.

**I 4 valori di `DIABETE3`** (dal codebook BRFSS 2015):

| valore | significato | n | tasso infarti |
|---|---|---|---|
| 1 | diabetico | 42.223 | 22,13% |
| 2 | diabete **solo in gravidanza** | 2.675 | 4,04% |
| 3 | **no** diabete | 277.049 | 6,73% |
| 4 | **prediabete / borderline** | 5.709 | **14,54%** |

Verifiche fatte sui dati, che confermano la codifica:
- `DIABETE3=2` → **100% femmine** (diabete solo in gravidanza) ✓;
- `DIABETE3=4` → tutti quelli che hanno risposto a `PREDIAB1` dicono "sì, mi hanno detto che ho
  il prediabete" (1.337 risposte, nessun "no") ✓.

**Conseguenze**:
- il valore 4 (prediabete, 14,5% di infarti) **non è "non diabetico"** (6,7%): per questo la
  `cleaning.py` mette `DIABAGE2`=0 (anni di diabete) per i valori {2,3,4} — i prediabetici non
  hanno il diabete, quindi 0 anni è corretto — e la categoria 0 di `INSULIN_CAT` va letta come
  "non diabetico *o* borderline";
- la distinzione resta comunque **disponibile al modello**, perché `DIABETE3` è una delle 129 colonne
  finali (mentre `INSULIN` non lo è più).

---

## 5. Fase 3 — Il taglio delle colonne ridondanti (153 → 129)

### 5.1 Come le abbiamo trovate (misurate, non a occhio)

Per ogni coppia di colonne chiediamo se una sia una **funzione deterministica** dell'altra: fissato il
valore di una, l'altra ha **sempre** lo stesso valore? Il *NaN* è contato come un valore a sé, quindi
si confronta anche il *pattern* dei mancanti. Chiamiamo **purezza** la frazione di righe in cui la
relazione vale: con purezza = 1,0000 la colonna è **letteralmente** ridondante.

Implementazione: `colonne_ridondanti(X, cols, min_righe=500, min_purezza=0.999)`.
Sulle 153 colonne abbiamo trovato **30 coppie**, tutte con purezza 1,0000.

### 5.2 Tre gruppi, tre trattamenti diversi

**(A) Copie (24 colonne) → buttate subito.** La figlia è deterministica dalla madre: buttarla
**non perde informazione**. Le 24 nomi (verificabili con la lista `DA_TOGLIERE_COPIE` del
notebook, che è la fonte di verità):

```
_RFHLTH   _RFHYPE5  _RFCHOL   _LTASTH1   _CASTHM1  _DRDXAR1  _EDUCAG   _INCOMG
_RFSMOK3  SMOKDAY2  _RFSEAT2  _RFSEAT3   _AIDTST3   _RACEG21  _RACEGR3  _RACE_G1
_AGE_G    _TOTINDA  _PA30021  _PAINDX1   _PASTAE1   HTIN4      _RFBMI5   FC60_
```

| famiglia | tengo | butto |
|---|---|---|
| salute generale | `GENHLTH` | `_RFHLTH` |
| ipertensione | `BPHIGH4` (+ `BPMEDS`) | `_RFHYPE5` |
| colesterolo | `TOLDHI2` | `_RFCHOL` |
| asma | `ASTHMA3`, `_ASTHMS1` | `_LTASTH1`, `_CASTHM1` |
| artrite | `HAVARTH3` | `_DRDXAR1` |
| fumo | `_SMOKER3`, `SMOKE100` | `SMOKDAY2`, `_RFSMOK3` |
| attività fisica | `EXERANY2`, `_PACAT1` | `_TOTINDA`, `_PA30021`, `_PAINDX1`, `_PASTAE1` |
| età | `_AGE80`, `_AGEG5YR` | `_AGE_G` |
| razza | `_RACE` | `_RACEG21`, `_RACEGR3`, `_RACE_G1` |
| reddito | `INCOME2` | `_INCOMG` |
| istruzione | `EDUCA` | `_EDUCAG` |
| cintura di sicurezza | `SEATBELT` | `_RFSEAT2`, `_RFSEAT3` |
| test HIV | `HIVTST6` | `_AIDTST3` |
| altezza | `HTM4` (metri) | `HTIN4` (pollici) |
| capacità funzionale | `MAXVO2_` | `FC60_` |
| BMI | `_BMI5` | `_RFBMI5` |

> **`_ASTHMS1` e `MAXVO2_` si tengono** (2 colonne in più, zero impatto sulla metrica): per loro il
> test automatico non trovava una "madre" singola, quindi per prudenza restano nel dataset.
> `_ASTHMS1` distingue "asma ancora attuale" da "asma in passato" (cosa che `ASTHMA3` non dice) e
> `MAXVO2_` (capacità funzionale) non è determinata da età/sex/BMI.

**(B) Quasi costanti (11 colonne) → aspettano la CV.** Non sono ridondanti: sono colonne che
**non variano** (spread < 0,3 punti percentuali, e per 4 di loro la differenza riguarda 39-70 righe
su 328.135).

| colonna | spread | classi | min classe | %NaN |
|---|---|---|---|---|
| `_FRT16` | 0,00 | 2 | 70 | 0,0% |
| `_VEG23` | 0,00 | 2 | 39 | 0,0% |
| `_FRUITEX` | 0,02 | 3 | 70 | 0,0% |
| `_VEGETEX` | 0,16 | 3 | 39 | 0,0% |
| `_FRTRESP` | 0,02 | 2 | 32.115 | 0,0% |
| `_VEGRESP` | 0,16 | 2 | 37.605 | 0,0% |
| `_MISFRTN` | 0,13 | 3 | 8.937 | 0,0% |
| `_MISVEGN` | 0,26 | 5 | 1.568 | 0,0% |
| `USENOW3` | 0,04 | 3 | 4.461 | 3,8% |
| `PAMISS1_` | 0,10 | 2 | 15.852 | 8,6% |
| `_HCVU651` | 0,12 | 2 | 21.999 | 35,9% |

> ⚠️ **spread basso ≠ inutile**: `ASYMPTOM` ha spread 6,95 ma lo 0,14% di supporto (77 infarti):
> quel gruppo è fortissimo. Per questo il criterio è "classe informativa ≥ 500 righe **e** spread
> basso". `_HCVU651` (35,9% di NaN) ha anche un problema aggiuntivo: il valore 9 significa
> "over 65, non applicabile" → va trattato come **categoria**, non come NaN da imputare.

**(C) Non sono ridondanti, solo parentele → non si toccano.** `CHOLCHK` (hai fatto il test?),
`BLOODCHO` (quanto?), `TOLDHI2` (ti hanno detto che è alto?), `_CHOLCHK` (ricodifica CDC) sono
**quattro domande diverse**. Idem `SMOKE100` vs `_SMOKER3`, `PHYSHLTH`/`MENTHLTH`/`POORHLTH`,
`DIABETE3`/`DIABAGE2`, `_RACE` vs `_PRACE1`/`_HISPANC`.

### 5.3 Perché buttare le copie (motivi tecnici, non estetici)

- **Collinearità**: `_RACE` + `_RACEG21` + `_RACEGR3` + `_RACE_G1` diventano 4 blocchi di variabili
  dummy identici → coefficienti instabili e difficili da interpretare (male per il report).
- **Leggibilità**: 153 → 129 colonne, molto più difendibili davanti a chi legge.
- **Dimensione**: 398 → ~336 feature dopo il taglio delle copie (meno variabili da manutenere).
- **Ridge (argomento secondario, e comunque da verificare in CV)**: due colonne identiche si
  *dividono* la penalità L2, quindi la penalità effettiva su quella direzione si dimezza. L'effetto
  reale dipende da λ, che tareremo in cross-validation: se il Δ di score non supera il rumore,
  il vantaggio è teorico e non lo useremo come giustificazione.

### 5.4 La coppia "difficile" (e perché non la decidiamo a mano)

`_BMI5` (continuo) e `_BMI5CAT` (4 classi) sono l'uno la discretizzazione dell'altro: sembrerebbe una
copia. Ma il rischio di infarto in funzione del BMI ha una **forma a U**, e per un modello *lineare*
la versione categorizzata può essere **migliore** di quella continua. Stesso discorso per l'età
(`_AGE80` continuo, `_AGEG5YR` a 13 classi, `_AGE65YR` binaria). Queste restano tutte e la scelta
la farà la CV (riga "grana diverse" della tabella di ablation).

### 5.5 Avvertenze sui controlli

- Il detector automatico (`colonne_ridondanti`) lavora solo su coppie **categoriche** con almeno
  500 righe per classe: per questo **non vede** `_EDUCAG` (la classe più rara di `EDUCA` ha 453
  righe), `_AGE_G` e `HTIN4` (coppie continue). Il notebook quindi fa un **controllo diretto**:
  per ogni copia buttata cerca una colonna **superstite** che la determina. Le tre "madre"
  superstiti risultano essere `EDUCA`, `_AGE80` e `HTM4` → nessuna perdita.
- `HTIN4` (pollici) e `HTM4` (**metri**, non centimetri!) sono la stessa grandezza in unità diverse:
  i valori coincidono, ma i pattern dei NaN differiscono su 1.353 righe (0,41%) → nessuna perdita
  (l'altezza in metri c'è comunque).
- `_ASTHMS1` e `MAXVO2_` **si tengono** (vedi sopra): 2 colonne in più, zero impatto sulla metrica.
- Un test "è determinata da tutto il resto del dataset?" con ~129 colonne superstiti è **vacuo**
  (ogni riga diventa quasi unica → il test passa sempre). Il test valido è "è determinata da **una
  o due** colonne superstiti?".

---

## 6. Stato finale

| | |
|---|---|
| `x_train` | (328.135, **129**) |
| `x_test` | (109.379, **129**) |
| NaN residui | 8,3% (train e test allineati) |
| colonne con > 60% NaN rimaste | 2: `DRADVISE`, `CVDASPRN` (le candidate contese) |
| `x_base` | (328.135, **127**) — modello di riferimento, senza le 2 candidate |
| target | `-1 / +1` → va convertito a `0 / 1` per la logistica |

**Cosa resta da fare prima di allenare** (preprocessing, da discutere insieme):

1. separare categoriche e continue;
2. imputare: **mediana + indicatore di missing** per le continue, **categoria "missing" +
   indicatore di missing** per le categoriche — **statistiche calcolate solo sul train**;
3. one-hot delle categoriche;
4. standardizzazione (z-score) delle continue — per la ridge la scala delle colonne conta;
5. target → {0, 1};
6. **la pipeline va rifatta dentro ogni fold della CV** (imputare con la mediana di tutto il dataset
   e poi fare la CV è *data leakage*).

---

## 7. Le funzioni di `cleaning.py`

| funzione | cosa fa |
|---|---|
| `read_header(path)` | legge la prima riga di un CSV e restituisce la lista dei nomi delle colonne |
| `_idx(cols)` | (interna) dizionario `{nome: indice}` per cercare colonne per nome |
| `fill_structural_zeros(X, cols)` | **passo 1**: mette 0 dove la domanda figlia non è stata posta perché la madre ha detto "no" (alcol, attività fisica, `POORHLTH`); trasforma `DIABAGE2` in "anni con il diabete" |
| `recode_none_to_zero(X, cols)` | **passo 2**: 88 / 888 / 555 (= "nessuno / mai") → 0 |
| `find_special_codes(X, cols)` | **passo 3a**: calcola (sul train) quali valori sono i codici "non so / rifiuta" di ogni colonna, guardando il massimo della colonna; esclude le continue (`NO_CODES`) e aggiunge i codici espliciti (`EXPLICIT_CODES`) |
| `special_codes_to_nan(X, cols, codes_by_col, verbose=False)` | **passo 3b**: sostituisce con NaN i codici trovati (stessi codici per train e test) |
| `clean(x_train, x_test, cols, verbose=False)` | esegue i tre passi nell'ordine giusto su train e test (i codici si calcolano sul train) |
| `informativita_colonne(x, y, names, max_cat=20, n_bin=10, top=None)` | per ogni colonna stampa `%NaN`, tasso di infarti fra chi ha NaN e fra chi ha risposto (`info_nan` = il NaN è informativo?), `spread` (quanto cambia il tasso fra i valori) e i conteggi degli infarti. Serve a capire *perché* tagliare |
| `identify_columns_to_drop(X_train, column_names, threshold)` | nomi delle colonne con proporzione di NaN sopra la soglia (0.6 → 123 colonne) |
| `drop_columns(X, cols, names)` | restituisce `(X senza le colonne indicate, lista dei nomi rimasti)` — da usare **sempre** su train e test insieme, con un `assert` che i nomi restanti coincidano |
| `colonne_ridondanti(X, cols, min_righe=500, min_purezza=0.999)` | trova le coppie in cui una colonna è una **funzione deterministica** dell'altra (NaN compreso) e restituisce `(madre, figlia, purezza)` |

In `helpers.py` (fornito dal corso): `load_csv_data(data_path, sub_sample=False)` e
`create_csv_submission(ids, y_pred, name)`.

---

## 8. Le 3 decisioni ancora aperte (tutte da misurare in CV)

1. **Le 11 colonne quasi costanti** → si applicano con una riga (`DA_TOGLIERE_VUOTE`).
2. **`CVDASPRN` / `DRADVISE`** → sono già isolate in `x_base` vs `x_train`: è l'ablation pronta.
3. **Le coppie "grana diverse"** (`_BMI5`/`_BMI5CAT`, età) → nessun taglio, solo scelta in CV.

**Il criterio con cui si decide** (serve anche per scrivere il report):

> La differenza di score tra due varianti **deve superare il rumore** della CV. Si confrontano i
> punteggi **fold per fold** (test appaiato, perché i fold sono gli stessi). Se |Δ| è più piccolo
> della deviazione standard tra i 5 fold → è rumore → si sceglie la variante **più semplice**.

Esempio (valori inventati, 5 fold sugli stessi pazienti, accuracy):

| variante | fold 1..5 | media | dev.std | Δ | verdetto |
|---|---|---|---|---|---|
| V0 = tutte | .78 .80 .76 .79 .77 | 0,780 | 0,016 | — | — |
| V1 = senza `DIAB` | .70 .72 .68 .71 .69 | 0,700 | 0,014 | **−0,080** | `DIAB` è fondamentale → la teniamo |
| V2 = senza `FUMO` | .77 .79 .75 .78 .76 | 0,770 | 0,014 | **−0,010** | Δ < rumore → la togliamo (stesso risultato, dataset più semplice) |
| V3 = V2 + `COLESTEROLO` | .83 .85 .81 .84 .82 | 0,830 | 0,016 | **+0,050** | 3× il rumore → la aggiungiamo |

---

## 9. Come riprodurre tutto

Il notebook **`notebooks/Giovanni.ipynb`** contiene l'intero processo, in quest'ordine (i numeri delle
celle cambiano a ogni modifica: cerca i blocchi per **titolo**).

| blocco (titolo della cella) | contenuto |
|---|---|
| *Caricamento dei dati (con cache)* | lettura con cache `.npy` in `cache/` (basta cancellare la cartella per rigenerarla) |
| *Colonne da eliminare a priori* | le 48 colonne eliminate + `drop_columns` |
| *Zeri strutturali (domande condizionali)* | `fill_structural_zeros` (con `DIABAGE2` → anni di diabete) |
| *Codici che significano "nessuno" → 0* | `recode_none_to_zero` |
| *Codici "non so / rifiuta" → NaN* | `find_special_codes` + `special_codes_to_nan` (codici calcolati sul train) |
| *Risultato* | quadro dei NaN residui + colonne vicine alla soglia |
| *Proviamo a togliere altre colonne* | `informativita_colonne` e `identify_columns_to_drop(0.6)` → 123 |
| *Perché tenere `INSULIN`, `CVDASPRN` e `DRADVISE`* | le evidenze numeriche sulla scelta delle 3 candidate |
| *Come ricostruiamo `INSULIN`* | `categorizza_insulin` → `INSULIN_CAT` e taglio delle 120 colonne |
| *Dove siamo arrivati* | riepilogo e costruzione di `x_base` |
| *Spiegazione delle colonne ridondanti* | rilevazione automatica + controlli (madre superstite per ogni copia) |
| *Applicazione del taglio* | taglio delle 24 copie + riepilogo finale |

Ogni cella di codice ha delle `print`, così l'avanzamento si legge eseguendo il notebook dall'alto.

> ⚠️ **Quale notebook usare**: **`notebooks/Giovanni.ipynb` è l'unico aggiornato** (ha
> `INSULIN_CAT` e il taglio delle colonne ridondanti). `pulizia_dataset.ipynb` contiene la *stessa
> pipeline in una versione vecchia*: **non usatelo**, oppure spostatelo in una cartella `old/`
> (consigliato, così i compagni non lo scelgono per sbaglio).

---

## 10. Avvertenze da tenere a mente

- **Il NaN è informativo**: non trattare "mancante" come "zero" senza motivo. È il motivo per cui
  esistono gli indicatori di missing.
- **Mai usare statistiche del test** (mediane, categorie one-hot, medie): tutto calcolato sul train.
- **Il taglio col 60% è una scelta, non una legge**: `BPMEDS` è al 59,9%, cioè *sulla soglia*
  (tenuta per 0,1 punti percentuali). Se in futuro si BUTTA `BPMEDS`, la sola variabile
  d'ipertensione che resta è `BPHIGH4` (4 classi; fra chi ha ipertensione il 16,3% ha avuto un
  infarto) → si perde la sotto-distinzione "prende i farmaci" (110.219 sì vs 21.366 no fra gli
  ipertesi, dato che `BPMEDS` è chiesto solo a chi ha `BPHIGH4`=1). `_RFHYPE5` è già stata buttata
  fra le copie, quindi non va recuperata.
- **`DIABETE3` ha 4 valori**: 1 = diabetico, 2 = diabete solo in gravidanza, 3 = no, 4 = prediabete
  / borderline. Il valore 4 (5.709 persone, 14,5% di infarti) **non va fuso** con il "no" (6,7%):
  la `cleaning.py` mette `DIABAGE2`=0 per {2,3,4} (corretto: i prediabetici non hanno il diabete) e
  la distinzione resta disponibile al modello perché `DIABETE3` è fra le 129 colonne finali.
- **`_HCVU651`**: il 9 = "over 65, non applicabile" → categoria, non NaN.
- **`CVDASPRN`/`DRADVISE` sono proxy post-outcome**: se la CV dice che aiutano, si usano e si
  dichiara nel report che sono "dopo l'evento", non cause.
- Il dataset è **sbilanciato 1:10**: la metrica della competizione (F1? accuracy?) va verificata
  prima di scegliere la soglia di classificazione (con 8,8% di positivi, la soglia 0,5 non è
  necessariamente quella giusta).
