# Progetto 1 ML: timeline e stato dei lavori

**Obiettivo:** predire l'infarto / malattia coronarica (`_MICHD`) dal dataset BRFSS 2015
**Scadenza:** 29 ottobre 2026 (codice su GitHub + report di 2 pagine in LaTeX + submission su AIcrowd)
**Vincoli:** solo Python standard, NumPy e matplotlib/seaborn (no pandas, no sklearn)

**Legenda:** ✅ fatto · 🔄 in corso · ⬜ da fare

---

## Timeline

| data | cosa | stato |
|---|---|---|
| 14/10 | Esplorazione iniziale: dimensioni, target, NaN | ✅ |
| 14/10 | Analisi delle cause dei NaN (4 gruppi) | ✅ |
| 14/10 | Lista delle colonne da eliminare | ✅ |
| 14/10 | `cleaning.py`: zeri strutturali + codici speciali | ✅ |
| 14/10 | Notebook `pulizia_dataset.ipynb` | ✅ |
| | Moduli opzionali: quali tenere | ⬜ |
| | Imputazione + one-hot + standardizzazione | ⬜ |
| | `implementations.py` (6 funzioni richieste) | ⬜ |
| | Baseline + cross-validation | ⬜ |
| | Miglioramenti (feature engineering, sbilanciamento, soglia) | ⬜ |
| | `run.py` riproducibile + submission finale | ⬜ |
| | Report (2 pagine) | ⬜ |

---

## 1. Esplorazione dei dati ✅

- [x] Dimensioni: train 328.135 × 321 feature (+ `Id`), test 109.379 righe
- [x] Target `_MICHD` ∈ {-1, 1}: **8,8% positivi**, quindi le classi sono sbilanciate (circa 1:10)
- [x] NaN: il 45% delle celle; 83 colonne senza NaN, 147 con più del 50% e 99 con più del 90%
- [x] Train e test hanno la stessa struttura di NaN (differenza massima 0,4%)

## 2. Cause dei NaN ✅

| gruppo | causa | n. colonne | decisione |
|---|---|---|---|
| 1 | Versione fisso / cellulare del questionario | ~18 | **eliminate** |
| 2 | Intervista interrotta (`DISPCODE`=1200, 15% delle righe) | ~40 | NaN veri → **imputazione** |
| 3 | Domande condizionali (solo se la risposta precedente è "sì") | ~45 | **0 dove è una quantità**, altrimenti categoria "non applicabile" |
| 4 | Moduli opzionali scelti dagli stati | ~100 | da decidere (vedi sezione 4) |

Scoperte importanti:
- [x] Il NaN è spesso **informativo**: per esempio il NaN di `DIABAGE2` vuol dire "non diabetico" (tasso di infarto 6,9% contro 22%)
- [x] **Leakage**: `HAREHAB1` è compilata solo per chi ha avuto un infarto (target = 1 nel 100% dei casi) → eliminata
- [x] **Codici speciali**: 7/77/777 = "non so", 9/99/999 = "rifiuta", 88/888 = "nessuno"

## 3. Pulizia (`cleaning.py` + `pulizia_dataset.ipynb`) ✅

- [x] **Eliminazione delle colonne** (321 → 273):
  - identificativi e date, gestione della chiamata, pesi statistici (la nostra lista `da_togliere`);
  - gruppo 1 (fisso/cellulare);
  - leakage (`HAREHAB1`);
  - colonne grezze ridondanti: `WEIGHT2`, `HEIGHT3` → `WTKG3`, `HTM4`, `_BMI5`; `EXEROFT*`, `EXERHMM*` → `PAFREQ*_`, `PADUR*_`; le date `FLSHTMY2`, `HIVTSTD3`
- [x] **Zeri strutturali**: 0 solo dove la domanda madre dice "no"
  - alcol (`AVEDRNK2`, `DRNK3GE5`, `MAXDRNKS`) ← `ALCDAY5` = 888
  - `POORHLTH` ← `PHYSHLTH` e `MENTHLTH` = 88
  - attività fisica 1 e 2 (`PA*`, `_MINAC*`, `METVL*`, `ACTIN*`) ← `EXERANY2` = 2, oppure `EXRACT21` = 88
  - `DIABAGE2` → **anni con il diabete** (0 per i non diabetici)
- [x] **Codici "nessuno" → 0**: 88 (giorni, figli, binge), 888 (`ALCDAY5`, `STRENGTH`), 555 (frutta e verdura)
- [x] **Codici "non so / rifiuta" → NaN**: regola sul valore massimo di ogni colonna, **calcolata sul train** e applicata identica al test
- [x] Testato su un campione di 3.000 righe
- [ ] Eseguire il notebook sul dataset completo

## 4. Moduli opzionali e NaN residui ⬜

- [ ] Decidere quali colonne dei moduli tenere. Candidate:
  - [ ] **`INSULIN`** (modulo diabete): tra i diabetici, chi usa insulina ha il 28% di infarti contro il 19% di chi non la usa. Va codificata con 4 categorie: non diabetico / sconosciuto / sì / no
  - [ ] `PREDIAB1`, `PDIABTST` (prediabete)
  - [ ] altre colonne del modulo diabete (`DOCTDIAB`, `CHKHEMO3`, …): verificarle in cross-validation
- [ ] Eliminare le altre colonne con più dell'X% di NaN (soglia 80–90%, da tarare in CV)
- [ ] Casi particolari:
  - [ ] `_HCVU651`: il 9 comprende gli over 65 (non applicabile), quindi va trattato come categoria e non come NaN
  - [ ] `INCOME2`: 17% di NaN dopo la pulizia, aggiungere un indicatore di missing
- [ ] Fondere le coppie madre/figlia categoriche (es. `BPHIGH4` + `BPMEDS` → non iperteso / iperteso senza farmaci / iperteso con farmaci: 3,7% / 7,6% / 18%)
- [ ] Preferire le variabili calcolate dal CDC (`_SMOKER3`, `_RFHYPE5`, `_RFCHOL`, `_TOTINDA`, …) ed eliminare le grezze corrispondenti

## 5. Preprocessing finale ⬜

- [ ] Separare le colonne categoriche dalle continue
- [ ] Imputazione: mediana per le continue, moda o categoria "missing" per le categoriche (**statistiche calcolate solo sul train**)
- [ ] Indicatori di missing per le colonne con molti NaN veri
- [ ] One-hot encoding delle categoriche
- [ ] Standardizzazione delle continue (media e deviazione standard del train)
- [ ] Target da {-1, 1} a {0, 1} per la regressione logistica
- [ ] Raccogliere tutto in `preprocessing.py`

## 6. Implementazione dei metodi ⬜

`implementations.py`: ogni funzione restituisce `(w, loss)`.
- [ ] `mean_squared_error_gd`
- [ ] `mean_squared_error_sgd` (batch size 1)
- [ ] `least_squares` (senza `np.linalg.lstsq`)
- [ ] `ridge_regression`
- [ ] `logistic_regression`
- [ ] `reg_logistic_regression` (la loss restituita non include la penalità)
- [ ] Passare i test ufficiali (`grading_tests` sul GitHub del corso)

## 7. Modellazione e valutazione ⬜

- [ ] k-fold cross-validation stratificata (k=5)
- [ ] Verificare la metrica della competizione su AIcrowd (F1?)
- [ ] **Baseline**: regressione logistica regolarizzata
- [ ] Tuning di λ, γ (learning rate) e `max_iters`
- [ ] Sbilanciamento: pesare la classe positiva oppure fare undersampling
- [ ] **Tarare la soglia di classificazione** per massimizzare l'F1
- [ ] Feature engineering: interazioni (età × ipertensione, età × diabete), termini polinomiali
- [ ] **Ablation study**: misurare quanto guadagna ogni passo di pulizia (servirà per il report)
- [ ] Prima submission su AIcrowd (massimo 5 al giorno)

## 8. Consegna ⬜

- [ ] `run.py` che riproduce la submission migliore da zero
- [ ] README nel repository GitHub (pubblico)
- [ ] Report LaTeX, al massimo 2 pagine più 1 per i riferimenti:
  - [ ] Introduzione e dataset
  - [ ] Pulizia: le 4 cause dei NaN, gli zeri strutturali, i codici speciali, il leakage
  - [ ] Metodi e cross-validation
  - [ ] Risultati: tabella di ablation e confronto tra modelli
  - [ ] Conclusioni
- [ ] Submission finale entro il **29/10**

---

## Decisioni prese (con motivazione)

| decisione | motivazione |
|---|---|
| Eliminare le colonne fisso/cellulare | Il NaN indica solo il tipo di telefono, non la salute |
| Eliminare `HAREHAB1` | Leakage: presente solo per chi ha già avuto un infarto |
| 0 solo dove la madre dice "no" | Se la madre è NaN il dato è davvero sconosciuto e va imputato, non messo a 0 |
| `DIABAGE2` → anni con il diabete | Lo 0 sull'età alla diagnosi non ha senso; gli anni di malattia sono predittivi (17% → 29%) |
| Codici speciali calcolati sul train | Garantisce la stessa trasformazione su train e test |
| Variabili calcolate dal CDC al posto delle grezze | Sono già pulite e in unità omogenee |

## Domande aperte

- [ ] Quale metrica usa AIcrowd? (F1 o accuracy?)
- [ ] Tenere `DISPCODE` come feature, visto che indica le interviste interrotte?
- [ ] Soglia sui NaN per eliminare le colonne: 80% o 90%?
