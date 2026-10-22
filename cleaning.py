"""
Pulizia dei codici speciali del BRFSS 2015 e riempimento degli "zeri strutturali".
Solo numpy (+ standard library), come richiesto dal progetto.

Uso tipico:
    cols = read_header("x_train.csv")          # nomi colonne (senza 'Id' se lo togliete prima)
    X = fill_structural_zeros(X, cols)          # 1) PRIMA: usa i codici grezzi delle domande madre
    X = recode_none_to_zero(X, cols)            # 2) 88 / 888 / 555 = "nessuno" -> 0
    codes = find_special_codes(X_train, cols)   # 3) 7/77/777, 9/99/999 = non so / rifiuta ...
    X = special_codes_to_nan(X, cols, codes)    #    ... -> NaN (codici calcolati sul train)
L'ordine conta: i passi 1-2 leggono i codici originali, il passo 3 li cancella.
"""
import numpy as np


def read_header(path):
    """Restituisce la lista dei nomi delle colonne di un csv."""
    with open(path) as f:
        return f.readline().strip().split(",")


def _idx(cols):
    return {c: i for i, c in enumerate(cols)}


# ---------------------------------------------------------------------------
# 1) Zeri strutturali: la domanda figlia (quantità) non è stata posta perché
#    la risposta alla madre è "no" -> la quantità vera è 0.
#    Si mette 0 SOLO dove la madre dice "no": se la madre è NaN (intervista
#    interrotta) la figlia resta NaN, perché lì il dato manca davvero.
# ---------------------------------------------------------------------------
NO_ALCOHOL = [("ALCDAY5", [888])]                       # 888 = nessun drink negli ultimi 30 gg
NO_EXERCISE = [("EXERANY2", [2])]                       # 2 = non fa attività fisica
NO_SECOND_ACTIVITY = [("EXERANY2", [2]), ("EXRACT21", [88])]  # 88 = nessuna seconda attività
NO_BAD_DAYS = "POORHLTH"                                # caso speciale (due madri, vedi sotto)

STRUCTURAL_ZEROS = {
    # alcol
    "AVEDRNK2": NO_ALCOHOL,   # drink medi per occasione
    "DRNK3GE5": NO_ALCOHOL,   # n. volte binge drinking
    "MAXDRNKS": NO_ALCOHOL,   # max drink in un'occasione
    # attività fisica principale (variabili calcolate dal CDC, unità già omogenee)
    "PAFREQ1_": NO_EXERCISE, "PADUR1_": NO_EXERCISE, "PAMIN11_": NO_EXERCISE,
    "PAVIG11_": NO_EXERCISE, "_MINAC11": NO_EXERCISE, "METVL11_": NO_EXERCISE,
    "ACTIN11_": NO_EXERCISE, "PA1MIN_": NO_EXERCISE, "PA1VIGM_": NO_EXERCISE,
    # seconda attività fisica
    "PAFREQ2_": NO_SECOND_ACTIVITY, "PADUR2_": NO_SECOND_ACTIVITY,
    "PAMIN21_": NO_SECOND_ACTIVITY, "PAVIG21_": NO_SECOND_ACTIVITY,
    "_MINAC21": NO_SECOND_ACTIVITY, "METVL21_": NO_SECOND_ACTIVITY,
    "ACTIN21_": NO_SECOND_ACTIVITY,
}


def fill_structural_zeros(X, cols):
    X = X.copy()
    ix = _idx(cols)
    for child, conditions in STRUCTURAL_ZEROS.items():
        if child not in ix:
            continue
        c = ix[child]
        mask = np.zeros(X.shape[0], dtype=bool)
        for parent, no_values in conditions:
            if parent in ix:
                mask |= np.isin(X[:, ix[parent]], no_values)
        X[mask & np.isnan(X[:, c]), c] = 0.0

    # POORHLTH (giorni in cui la salute ha impedito le attività) è chiesta solo
    # se PHYSHLTH o MENTHLTH > 0: se entrambe sono 88 ("nessun giorno") -> 0.
    if {"POORHLTH", "PHYSHLTH", "MENTHLTH"} <= ix.keys():
        p = ix["POORHLTH"]
        mask = (X[:, ix["PHYSHLTH"]] == 88) & (X[:, ix["MENTHLTH"]] == 88)
        X[mask & np.isnan(X[:, p]), p] = 0.0

    # DIABAGE2 (età alla diagnosi) -> anni con il diabete; 0 per i non diabetici.
    if {"DIABAGE2", "DIABETE3", "_AGE80"} <= ix.keys():
        d = ix["DIABAGE2"]
        age_dx = X[:, d].copy()
        age_dx[age_dx >= 98] = np.nan                       # 98 = non so, 99 = rifiuta
        years = np.clip(X[:, ix["_AGE80"]] - age_dx, 0, None)
        non_diab = np.isin(X[:, ix["DIABETE3"]], [2, 3, 4])  # no / solo in gravidanza / prediabete
        years[non_diab] = 0.0
        X[:, d] = years                                      # la colonna ora significa "anni con diabete"
    return X


# ---------------------------------------------------------------------------
# 2) Codici che significano "nessuno / mai" -> 0
# ---------------------------------------------------------------------------
NONE_CODES = {
    "PHYSHLTH": [88], "MENTHLTH": [88], "POORHLTH": [88],   # giorni
    "CHILDREN": [88],                                       # n. figli
    "DRNK3GE5": [88],                                       # n. binge
    "ALCDAY5": [888],                                       # giorni con alcol
    "STRENGTH": [888],                                      # allenamenti forza
    "FRUITJU1": [555], "FRUIT1": [555], "FVBEANS": [555],   # frutta/verdura grezze
    "FVGREEN": [555], "FVORANG": [555], "VEGETAB1": [555],
}


def recode_none_to_zero(X, cols):
    X = X.copy()
    ix = _idx(cols)
    for col, codes in NONE_CODES.items():
        if col in ix:
            c = ix[col]
            X[np.isin(X[:, c], codes), c] = 0.0
    return X


# ---------------------------------------------------------------------------
# 3) Codici "non so" / "rifiuta" -> NaN
# ---------------------------------------------------------------------------
# Colonne continue (calcolate dal CDC o pesi): non hanno codici, non toccarle.
NO_CODES = {
    "_BMI5", "WTKG3", "HTM4", "HTIN4", "_AGE80", "_FRUTSUM", "_VEGESUM",
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_",
    "METVL11_", "METVL21_", "PADUR1_", "PADUR2_", "PAMIN11_", "PAMIN21_",
    "PA1MIN_", "PAVIG11_", "PAVIG21_", "PA1VIGM_", "_MINAC11", "_MINAC21",
    "STRFREQ_", "PAFREQ1_", "PAFREQ2_",
    "DIABAGE2",  # dopo fill_structural_zeros contiene gli anni con diabete
    "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_CLLCPWT", "_DUALCOR",
}
# Colonne con codici "fuori schema" (dal codebook BRFSS 2015).
EXPLICIT_CODES = {
    "_AGEG5YR": [14],        # 14 = non so / rifiuta
    "_AGE65YR": [3],
    "DROCDY3_": [900],
    "_DRNKWEK": [99900],
    "ASERVIST": [98], "ASDRVIST": [98],
}


def find_special_codes(X, cols):
    """
    Calcola (sul TRAIN) quali valori sono codici "non so / rifiuta" in ogni colonna.
    Regola, per ogni colonna non continua, guardando il valore massimo:
      - max = 9, 99, 999, 9999, ...  -> i codici sono 9..9 (rifiuta) e 7..7 (non so)
        (con una cifra sola, il 7 si tratta come codice solo se il 6 non compare:
         es. EMPLOY1 ha 7 = "pensionato", che è una risposta valida)
      - max = 7 con il 6 assente     -> 7 è "non so" (nessuno ha rifiutato)
    Più i codici espliciti in EXPLICIT_CODES.
    Restituisce un dizionario {nome_colonna: [codici]}.
    """
    codes_by_col = {}
    for j, name in enumerate(cols):
        if name in NO_CODES:
            continue
        col = X[:, j]
        vals = np.unique(col[~np.isnan(col)])
        if vals.size == 0:
            continue
        codes = list(EXPLICIT_CODES.get(name, []))
        vmax = vals.max()
        if vmax == int(vmax) and vmax >= 9 and set(str(int(vmax))) == {"9"}:
            n = len(str(int(vmax)))
            codes.append(int("9" * n))
            if n > 1 or 6 not in vals:
                codes.append(int("7" * n))
        elif vmax == 7 and 6 not in vals:
            codes.append(7)
        if codes:
            codes_by_col[name] = sorted(set(codes))
    return codes_by_col


def special_codes_to_nan(X, cols, codes_by_col, verbose=False):
    """Sostituisce con NaN i codici trovati da find_special_codes (stessi codici per train e test)."""
    X = X.copy()
    for j, name in enumerate(cols):
        if name not in codes_by_col:
            continue
        hit = np.isin(X[:, j], codes_by_col[name])
        X[hit, j] = np.nan
        if verbose and hit.any():
            print(f"{name:10s} codici {codes_by_col[name]} -> NaN su {int(hit.sum())} righe")
    return X


def clean(x_train, x_test, cols, verbose=False):
    """Applica i tre passi nell'ordine corretto a train e test (codici calcolati sul train)."""
    x_train = recode_none_to_zero(fill_structural_zeros(x_train, cols), cols)
    x_test = recode_none_to_zero(fill_structural_zeros(x_test, cols), cols)
    codes = find_special_codes(x_train, cols)
    return (special_codes_to_nan(x_train, cols, codes, verbose),
            special_codes_to_nan(x_test, cols, codes))


def informativita_colonne(x, y, names, max_cat=20, n_bin=10, top=None):
    """
    x: array 2D (con NaN), y: target in {-1, 1}, names: nomi colonne.
    Per ogni colonna calcola:
      pct_nan    : % di NaN
      rate_nan   : % di infarti tra i NaN
      rate_val   : % di infarti tra chi ha risposto
      info_nan   : |rate_nan - rate_val|  (il NaN è informativo?)
      spread     : quanto cambia il tasso di infarto tra i valori (i valori sono informativi?)
      pos_nan, pos_val: n. di infarti nei due gruppi (per capire se è rumore)
    """
    y01 = (y == 1).astype(float)
    righe = []

    for j, name in enumerate(names):
        col = x[:, j]
        nan_mask = np.isnan(col)
        n_nan = nan_mask.sum()
        n_val = len(col) - n_nan
        if n_val == 0:
            continue

        # 1) informazione del NaN
        rate_val = y01[~nan_mask].mean()
        rate_nan = y01[nan_mask].mean() if n_nan > 0 else np.nan
        info_nan = abs(rate_nan - rate_val) if n_nan > 0 else 0.0

        # 2) informazione dei valori, solo tra chi ha risposto
        v = col[~nan_mask]
        yv = y01[~nan_mask]
        if len(np.unique(v)) > max_cat:
            # colonna continua: la spezzo in n_bin gruppi di uguale numerosità
            bordi = np.unique(np.quantile(v, np.linspace(0, 1, n_bin + 1)[1:-1]))
            gruppi = np.searchsorted(bordi, v, side="right")
        else:
            _, gruppi = np.unique(v, return_inverse=True)

        n_g = np.bincount(gruppi)
        pos_g = np.bincount(gruppi, weights=yv)
        ok = n_g > 0
        rate_g = pos_g[ok] / n_g[ok]
        spread = np.sum(n_g[ok] / n_val * np.abs(rate_g - rate_val))

        righe.append((name, 100 * n_nan / len(col), 100 * rate_nan, 100 * rate_val,
                      100 * info_nan, 100 * spread,
                      int(y01[nan_mask].sum()), int(yv.sum())))

    righe.sort(key=lambda r: max(r[4], r[5]), reverse=True)
    if top:
        righe = righe[:top]

    print(f"{'colonna':<12}{'%NaN':>7}{'rate_nan':>10}{'rate_val':>10}"
          f"{'info_nan':>10}{'spread':>8}{'pos_nan':>9}{'pos_val':>9}")
    for r in righe:
        print(f"{r[0]:<12}{r[1]:7.1f}{r[2]:10.1f}{r[3]:10.1f}{r[4]:10.1f}{r[5]:8.1f}{r[6]:9d}{r[7]:9d}")
    return righe


def identify_columns_to_drop(X_train, column_names, threshold):
    """
    Restituisce i nomi delle colonne con proporzione di NaN sopra threshold.

    Parametri:
        X_train (np.ndarray): dati di training (già puliti con clean()).
        column_names (list): nomi delle colonne, nello stesso ordine di X_train.
        threshold (float): frazione tra 0 e 1 (es. 0.6 = 60% di NaN).
    Restituisce:
        list: nomi delle colonne da rimuovere.
    """
    assert X_train.shape[1] == len(column_names), "colonne e nomi non coincidono"

    nan_proportion = np.isnan(X_train).mean(axis=0)       # un valore tra 0 e 1 per colonna
    idx = np.where(nan_proportion > threshold)[0]         # indici delle colonne sopra soglia
    return np.array(column_names)[idx].tolist()

def drop_columns(X, cols, names):
    """
    Restituisce (X senza le colonne indicate, lista dei nomi delle colonne rimaste).

    Args:
        X    (np.ndarray): matrice 2D (n_righe, n_colonne)
        cols (list): nomi delle colonne, nello stesso ordine di X
        names (list): nomi delle colonne da eliminare
    Returns:
        (X_new, cols_new)
    """
    to_remove = set(names)
    keep = [j for j, c in enumerate(cols) if c not in to_remove]
    return X[:, keep], [cols[j] for j in keep]



def colonne_ridondanti(X, cols, min_righe=500, min_purezza=0.999):
    """
    Trova le coppie di colonne in cui una e' una funzione deterministica dell'altra.
    X: array 2D (con NaN), cols: nomi delle colonne.
    Restituisce una lista di (madre, figlia, purezza).
    """
    def valori(j):
        v = X[:, j]; m = ~np.isnan(v)
        u, c = np.unique(v[m], return_counts=True)
        return u, c, m.sum()
    def codici(j):
        _, inv = np.unique(X[:, j], return_inverse=True)
        return inv.astype(np.int64), len(np.unique(X[:, j]))
    def purezza(a_codes, na, b_codes, nb):
        key = a_codes * nb + b_codes
        uniq, cnt = np.unique(key, return_counts=True)
        a_of = uniq // nb
        tot = np.zeros(na); np.add.at(tot, a_of, cnt)
        mx = np.zeros(na); np.maximum.at(mx, a_of, cnt)
        return float((mx / np.maximum(tot, 1)).min())

    out = []
    for a in range(len(cols)):
        ua, ca, na_ = valori(a)
        if na_ < min_righe or len(ua) < 2 or ca.min() < min_righe:
            continue
        for b in range(a + 1, len(cols)):
            ub, cb, nb_ = valori(b)
            if nb_ < min_righe or len(ub) < 2 or cb.min() < min_righe:
                continue
            if len(ua) > 25 or len(ub) > 25:      # continue: gestite a parte
                continue
            ca_codes, nca = codici(a); cb_codes, ncb = codici(b)
            p = purezza(ca_codes, nca, cb_codes, ncb)
            if p >= min_purezza:
                out.append((cols[a], cols[b], p))   # figlia = b
            else:
                p = purezza(cb_codes, ncb, ca_codes, nca)
                if p >= min_purezza:
                    out.append((cols[b], cols[a], p))  # figlia = a
    return out


# =============================================================================
# PARTE 2 - PREPROCESSING (portato da notebooks/notebookmatteo.ipynb)
# =============================================================================
# Tutte le funzioni qui sotto lavorano su (X, cols) e restituiscono (X, cols):
# sono pensate per essere applicate a train e test con gli stessi identici
# argomenti, cosi' che le due matrici restino allineate.
#
# Regola d'oro (vale per tutta la parte 2):
#   - le "regole fisse" (nessuna statistica) si applicano UNA volta a train e test;
#   - le "statistiche" (mediane, livelli, media, deviazione standard) si imparano
#     SOLO sul train con fit_* e si applicano identiche con transform_*;
#     nella cross-validation, fit_prep va chiamata DENTRO ogni fold.
#
# Ordine di esecuzione (uguale a quello del notebook):
#   1a rimuovi_grezze          -> toglie le 8 colonne con codifica mista
#   1b trasforma_imfv         -> IMFVPLAC diventa IMFV_LAVORO; tolte EXRACT11/21
#   1c rimuovi_hcvu651        -> toglie _HCVU651 (redundante con HLTHPLN1 + eta')
#   2  controlli_gruppo2 + trasforma_gruppo2 -> NaN strutturali -> "no"
#   3  rimuovi_gruppo3        -> tolte le colonne doppie (purezza + one-hot)
#   4  regole_fisse_gruppo4   -> tolte le somme esatte, valori impossibili -> NaN
#   5  correggi_collinearita  -> INSULIN_CAT, SMOKE100, artrite, ASTHMA3
#   poi: fit_prep / transform_prep -> la matrice finale (bias + continue + categoriche)
# =============================================================================

SOGLIA_PUREZZA = 0.999     # 1.0000 = la figlia e' una funzione esatta della madre
CODICE_NAN = -1            # valore usato al posto del NaN prima del one-hot
USA_LOG = True             # ⚠️ scelta da confrontare in CV (log(1+x) sulle code lunghe)

# Le 38 colonne CONTINUE (quantita'). Tutto il resto e' categorico.
# Dopo il gruppo 4 restano 34: i 4 totali (_FRUTSUM, _VEGESUM, PA1MIN_, PA1VIGM_)
# vengono tolti perche' sono somme esatte delle altre.
CONTINUE = {
    '_AGE80', 'CHILDREN', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'DIABAGE2',
    'WTKG3', 'HTM4', '_BMI5', 'MAXVO2_',
    'AVEDRNK2', 'DRNK3GE5', 'MAXDRNKS', 'DROCDY3_', '_DRNKWEK',
    'FTJUDA1_', 'FRUTDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_',
    '_FRUTSUM', '_VEGESUM',
    'STRFREQ_', 'PAFREQ1_', 'PAFREQ2_', 'PADUR1_', 'PADUR2_',
    'PAMIN11_', 'PAMIN21_', 'PA1MIN_', 'PAVIG11_', 'PAVIG21_', 'PA1VIGM_',
    '_MINAC11', '_MINAC21', 'METVL11_', 'METVL21_',
}

# Colonne con coda lunga: su queste si applica log(1+x) prima dello z-score.
LOG_COLS = {
    "AVEDRNK2", "DRNK3GE5", "MAXDRNKS", "_DRNKWEK",                              # alcol
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_",      # frutta e verdura
    "PADUR1_", "PADUR2_", "PAFREQ1_", "PAFREQ2_", "STRFREQ_",                    # attivita' fisica
    "_MINAC11", "_MINAC21", "PAMIN11_", "PAMIN21_", "PAVIG11_", "PAVIG21_",
}

# Valori fisicamente impossibili: oltre il limite -> NaN (non e' un estremo reale,
# e' un errore di compilazione o un codice rimasto nei dati).
LIMITI = {
    "CHILDREN": 15,
    "AVEDRNK2": 30, "DRNK3GE5": 30, "MAXDRNKS": 40, "_DRNKWEK": 14000,
    "FTJUDA1_": 16, "FRUTDA1_": 16,
    "BEANDAY_": 23, "GRENDAY_": 23, "ORNGDAY_": 23, "VEGEDA1_": 23,
    "PAFREQ1_": 28, "PAFREQ2_": 28, "STRFREQ_": 28,
    "_MINAC11": 10080, "_MINAC21": 10080, "PAMIN11_": 10080, "PAMIN21_": 10080,
    "PAVIG11_": 10080, "PAVIG21_": 10080,
}

# Somme esatte delle altre colonne -> collinearita' esatta -> da togliere.
TOTALI = ["_FRUTSUM", "_VEGESUM", "PA1MIN_", "PA1VIGM_"]

# Gruppo 3A: copia esatta di _PACAT1 + gli 8 flag di servizio su frutta/verdura.
DA_TOGLIERE_3A = ['_PA300R2',
                  '_FRT16', '_VEG23', '_FRUITEX', '_VEGETEX',
                  '_FRTRESP', '_VEGRESP', '_MISFRTN', '_MISVEGN']

# Gruppo 3B: la razza si riduce a _RACE; _PA150R2 e' un raggruppamento di _PACAT1.
DA_TOGLIERE_3B = ['_HISPANC', '_PRACE1', '_MRACE1', '_PA150R2']

# Codici di IMFVPLAC che significano "vaccino al lavoro" o "a scuola" (codebook 2015).
LUOGHI_LAVORO_SCUOLA = [8, 11]

# Nelle _LMT* la categoria "non ha l'artrite" (HAVARTH3 = 2) va ricodificata in
# "non lo limita": _LMTACT1/_LMTWRK1 3 -> 2, _LMTSCL1 4 -> 3.
NO_ARTRITE = {"_LMTACT1": 3, "_LMTWRK1": 3, "_LMTSCL1": 4}


def purezza(a, b):
    """
    Frazione di righe in cui b assume il valore piu' frequente associato al valore di a.
    1.0  -> b e' una funzione esatta di a.
    Il NaN e' trattato come un valore a se'.
    """
    a = np.where(np.isnan(a), -999.0, a)
    b = np.where(np.isnan(b), -999.0, b)
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    nb = ib.max() + 1
    coppie, conte = np.unique(ia.astype(np.int64) * nb + ib, return_counts=True)
    gruppo_a = coppie // nb
    migliore = np.zeros(ia.max() + 1, dtype=np.int64)
    np.maximum.at(migliore, gruppo_a, conte)     # per ogni valore di a: la coppia piu' frequente
    return migliore.sum() / len(a)


def fasce_eta(age):
    """Fasce di _AGEG5YR: 1 = 18-24, 2 = 25-29, ..., 12 = 75-79, 13 = 80+."""
    return np.where(age < 25, 1, np.minimum(13, (age - 25) // 5 + 2))


# ---------------------------------------------------------------------------
# Gruppo 1a - colonne grezze con codifica mista (frequenza + unita' nel numero)
# ---------------------------------------------------------------------------
COPPIE_GREZZA_PULITA = {
    'ALCDAY5':  'DROCDY3_',
    'STRENGTH': 'STRFREQ_',
    'FRUITJU1': 'FTJUDA1_',
    'FRUIT1':   'FRUTDA1_',
    'FVBEANS':  'BEANDAY_',
    'FVGREEN':  'GRENDAY_',
    'FVORANG':  'ORNGDAY_',
    'VEGETAB1': 'VEGEDA1_',
}


def rimuovi_grezze(X, cols, coppie=None, soglia=SOGLIA_PUREZZA, verbose=False):
    """
    Toglie le colonne grezze la cui versione "pulita" del CDC ne e' una funzione
    esatta (purezza >= soglia): non si perde informazione e si evita che un
    modello lineare legga "107 (7 giorni a settimana)" come minore di
    "201 (1 giorno al mese)".

    Ritorna (X, cols, report) con report = lista di dizionari.
    """
    coppie = COPPIE_GREZZA_PULITA if coppie is None else coppie
    ix = _idx(cols)
    report, da_togliere = [], []
    for grezza, pulita in coppie.items():
        if grezza not in ix:
            report.append({"grezza": grezza, "pulita": pulita, "esito": "gia' tolta"})
            continue
        if pulita not in ix:
            report.append({"grezza": grezza, "pulita": pulita, "esito": "manca la versione pulita"})
            continue
        a, b = X[:, ix[grezza]], X[:, ix[pulita]]
        p = purezza(a, b)
        solo_a = int((np.isnan(a) & ~np.isnan(b)).sum())
        solo_b = int((~np.isnan(a) & np.isnan(b)).sum())
        ok = p >= soglia
        if ok:
            da_togliere.append(grezza)
        report.append({"grezza": grezza, "pulita": pulita, "purezza": p,
                       "nan_solo_grezza": solo_a, "nan_solo_pulita": solo_b,
                       "esito": "TOLTA" if ok else "RESTA (da guardare a mano)"})
        if verbose:
            print(f"   {grezza:10s} {pulita:10s} {p:9.4f}  {solo_a:8d} {solo_b:8d}  "
                  f"{'TOLTA' if ok else 'RESTA'}")
    X, cols = drop_columns(X, cols, da_togliere)
    if verbose:
        print(f"   tolte {len(da_togliere)} colonne: {da_togliere}")
    return X, cols, report


# ---------------------------------------------------------------------------
# Gruppo 1b - IMFVPLAC -> IMFV_LAVORO ; tolte EXRACT11 / EXRACT21
# ---------------------------------------------------------------------------
def trasforma_imfv(X, cols):
    """
    IMFVPLAC (dove ha fatto il vaccino) -> IMFV_LAVORO (1 = al lavoro o a scuola).
    Sono tolte EXRACT11/EXRACT21 (tipo di attivita': 76-77 categorie, il tipo e'
    soprattutto un indicatore dell'eta'; l'intensita' separa di piu').

    Regola fissa: usa solo i valori della riga, quindi si applica identica a
    train e test.
    """
    if "IMFVPLAC" in cols:
        lav = np.isin(X[:, cols.index("IMFVPLAC")], LUOGHI_LAVORO_SCUOLA).astype(float)
        X, cols = drop_columns(X, cols, ["IMFVPLAC"])
        X = np.column_stack([X, lav])
        cols = cols + ["IMFV_LAVORO"]
    return drop_columns(X, cols, ["EXRACT11", "EXRACT21"])


# ---------------------------------------------------------------------------
# Gruppo 1c - _HCVU651 (copertura sanitaria 18-64 anni)
# ---------------------------------------------------------------------------
def rimuovi_hcvu651(X, cols, soglia=SOGLIA_PUREZZA, verbose=False):
    """
    _HCVU651 e' determinata da HLTHPLN1 + eta' (purezza 1,0000): toglierla non
    perde informazione. Inoltre il suo NaN mescolava "over 65" e "non so".
    Ritorna (X, cols, report).
    """
    ix = _idx(cols)
    if "_HCVU651" not in ix:
        return X, cols, [{"colonna": "_HCVU651", "esito": "gia' tolta"}]

    hcv = X[:, ix["_HCVU651"]]
    hlth = np.nan_to_num(X[:, ix["HLTHPLN1"]], nan=9)       # NaN di HLTHPLN1 = valore a se'
    chiavi = {}
    if "_AGE65YR" in ix:
        age65 = np.nan_to_num(X[:, ix["_AGE65YR"]], nan=9)  # 1 = 18-64, 2 = 65+, 9 = non dichiarata
        chiavi["HLTHPLN1 + _AGE65YR"] = hlth * 10 + age65
    if "_AGE80" in ix:
        chiavi["HLTHPLN1 + (_AGE80 >= 65)"] = hlth * 10 + (X[:, ix["_AGE80"]] >= 65)

    risultati = {nome: purezza(k, hcv) for nome, k in chiavi.items()}
    if verbose:
        for nome, p in risultati.items():
            print(f"   {nome:28s} purezza {p:9.4f}")
    if risultati and max(risultati.values()) >= soglia:
        X, cols = drop_columns(X, cols, ["_HCVU651"])
        esito = "TOLTA"
    else:
        esito = "RESTA"
    return X, cols, [{"colonna": "_HCVU651", "purezze": risultati, "esito": esito}]


# ---------------------------------------------------------------------------
# Gruppo 2 - NaN strutturali (domande poste solo a chi ha risposto "sì" alla madre)
# ---------------------------------------------------------------------------
def controlli_gruppo2(X, cols, verbose=False):
    """
    Controlli PRELIMINARI (non modificano nulla): restituisce il dizionario dei
    controlli passati, da usare come argomento di trasforma_gruppo2.
    """
    ix = _idx(cols)
    col = lambda n: X[:, ix[n]]
    ok = {}

    def controllo_condizionale(figlia, madre, si):
        if figlia not in ix or madre not in ix:
            return False                      # la domanda (o la madre) non c'e' piu'
        f, m = col(figlia), col(madre)
        risposte = ~np.isnan(f)
        fuori = risposte & ~np.isnan(m) & ~np.isin(m, si)   # ha risposto ma la madre non e' "sì"
        da_riempire = np.isnan(f) & ~np.isnan(m) & ~np.isin(m, si)
        passa = fuori.sum() <= 0.001 * risposte.sum()
        if verbose:
            print(f"   {figlia:8s} (madre {madre}): risposte con madre != sì: {fuori.sum():5d}   "
                  f"NaN da riempire: {da_riempire.sum():6d}  -> {'OK' if passa else 'NON PASSA'}")
        return passa

    ok["BPMEDS"] = controllo_condizionale("BPMEDS", "BPHIGH4", [1])
    ok["TOLDHI2"] = controllo_condizionale("TOLDHI2", "BLOODCHO", [1])
    ok["CHOLCHK"] = controllo_condizionale("CHOLCHK", "BLOODCHO", [1])

    if "CHOLCHK" in ix:
        valori = np.unique(col("CHOLCHK")[~np.isnan(col("CHOLCHK"))])
        if not np.array_equal(valori, [1, 2, 3, 4]):
            ok["CHOLCHK"] = False
    else:
        ok["CHOLCHK"] = False

    ok["_CHOLCHK"] = ("_CHOLCHK" in ix and "BLOODCHO" in ix and "CHOLCHK" in ix
                      and purezza(np.nan_to_num(col("BLOODCHO"), nan=9) * 10
                                  + np.nan_to_num(col("CHOLCHK"), nan=9),
                                  col("_CHOLCHK")) >= SOGLIA_PUREZZA)

    if "_AGEG5YR" in ix and "_AGE80" in ix:
        g5 = fasce_eta(col("_AGE80"))
        presenti = ~np.isnan(col("_AGEG5YR"))
        accordo = float(np.mean(g5[presenti] == col("_AGEG5YR")[presenti]))
        ok["_AGEG5YR"] = accordo >= SOGLIA_PUREZZA
    else:
        ok["_AGEG5YR"] = False

    ok["_AGE65YR"] = ("_AGE65YR" in ix and "_AGEG5YR" in ix
                      and purezza(col("_AGEG5YR"), col("_AGE65YR")) >= SOGLIA_PUREZZA)
    return ok


def trasforma_gruppo2(X, cols, ok):
    """
    Applica il gruppo 2 SOLO dove il controllo e' passato:
      - BPMEDS : NaN con BPHIGH4 in {2,3,4} -> 2 ("non prende farmaci")
      - TOLDHI2: NaN con BLOODCHO = 2       -> 2 ("non glielo hanno mai detto")
      - CHOLCHK: NaN con BLOODCHO = 2       -> 4 ("5 anni o piu' / mai")
      - _AGEG5YR: NaN                       -> fascia ricavata da _AGE80
      - tolte: _CHOLCHK, _AGE65YR
    """
    X = X.copy()
    j = _idx(cols)
    if ok.get("BPMEDS") and "BPMEDS" in j and "BPHIGH4" in j:
        m = np.isnan(X[:, j["BPMEDS"]]) & np.isin(X[:, j["BPHIGH4"]], [2, 3, 4])
        X[m, j["BPMEDS"]] = 2
    if "BLOODCHO" in j:
        mai_testato = X[:, j["BLOODCHO"]] == 2
        if ok.get("TOLDHI2") and "TOLDHI2" in j:
            m = np.isnan(X[:, j["TOLDHI2"]]) & mai_testato
            X[m, j["TOLDHI2"]] = 2
        if ok.get("CHOLCHK") and "CHOLCHK" in j:
            m = np.isnan(X[:, j["CHOLCHK"]]) & mai_testato
            X[m, j["CHOLCHK"]] = 4
    if ok.get("_AGEG5YR") and "_AGEG5YR" in j and "_AGE80" in j:
        m = np.isnan(X[:, j["_AGEG5YR"]])
        X[m, j["_AGEG5YR"]] = fasce_eta(X[m, j["_AGE80"]])
    da_togliere = [c for c in ["_CHOLCHK", "_AGE65YR"] if ok.get(c)]
    return drop_columns(X, cols, da_togliere)


# ---------------------------------------------------------------------------
# Gruppo 3 - colonne categoriche doppie
# ---------------------------------------------------------------------------
def coppie_categoriche_ridondanti(X, cols, continue_cols=None, soglia_purezza=SOGLIA_PUREZZA,
                                  soglia_quota=0.99, verbose=True):
    """
    Cerca fra TUTTE le coppie di colonne categoriche quelle in cui una e' una
    funzione deterministica dell'altra. Per non farsi ingannare dalle colonne
    quasi costanti si chiede anche che la madre spieghi la variabilita' della figlia:

        frazione spiegata = (purezza - quota della categoria piu' frequente) / (1 - quota)

    Ritorna una lista di (figlia, madre, purezza, spiegata). Non modifica nulla.
    """
    continue_cols = CONTINUE if continue_cols is None else continue_cols
    cat_cols = [c for c in cols if c not in continue_cols]
    n = X.shape[0]

    codici, ncat, quota_max = {}, {}, {}
    for c in cat_cols:
        v = X[:, cols.index(c)]
        _, inv, cnt = np.unique(np.where(np.isnan(v), -999.0, v),
                                return_inverse=True, return_counts=True)
        codici[c], ncat[c], quota_max[c] = inv.astype(np.int64), len(cnt), cnt.max() / len(v)

    if verbose:
        print(f"   categoriche: {len(cat_cols)}   continue: {len(cols) - len(cat_cols)}")
        for c in cat_cols:
            if ncat[c] > 20:
                print(f"   ATTENZIONE: {c} ha {ncat[c]} valori distinti: e' davvero categorica?")

    trovate = []
    for a in cat_cols:
        for b in cat_cols:
            if a == b or ncat[b] == 1:
                continue
            tab = np.bincount(codici[a] * ncat[b] + codici[b],
                              minlength=ncat[a] * ncat[b]).reshape(ncat[a], ncat[b])
            p = tab.max(axis=1).sum() / n           # per ogni valore di A, il B piu' frequente
            spiegata = (p - quota_max[b]) / (1 - quota_max[b])
            if p >= soglia_purezza and spiegata >= soglia_quota:
                trovate.append((b, a, p, spiegata))
    if verbose:
        coppie = {(b, a) for b, a, _, _ in trovate}
        print(f"   coppie trovate: {len(trovate)}   (figlia = funzione esatta della madre)")
        for b, a, p, s in sorted(trovate):
            nota = "RECIPROCA" if (a, b) in coppie else ""
            print(f"   {b:10s} {a:10s} {p:8.4f} {s:9.4f}  {nota}")
    return trovate


def rimuovi_gruppo3(X, cols, da_togliere_3a=None, da_togliere_3b=None):
    """Tolte le colonne doppie trovate dai controlli di purezza (3A) e nel one-hot (3B)."""
    da_togliere_3a = DA_TOGLIERE_3A if da_togliere_3a is None else da_togliere_3a
    da_togliere_3b = DA_TOGLIERE_3B if da_togliere_3b is None else da_togliere_3b
    return drop_columns(X, cols, list(da_togliere_3a) + list(da_togliere_3b))


# ---------------------------------------------------------------------------
# Gruppo 4 - continue: somme esatte, valori impossibili, mediana, log, z-score
# ---------------------------------------------------------------------------
def valori_impossibili_a_nan(X, cols):
    """Valori oltre i limiti fisici (LIMITI) -> NaN; DIABAGE2 < 0 o > eta' -> NaN."""
    X = X.copy()
    j = _idx(cols)
    for c, lim in LIMITI.items():
        if c in j:
            m = X[:, j[c]] > lim                    # i NaN danno False
            X[m, j[c]] = np.nan
    if "DIABAGE2" in j and "_AGE80" in j:
        d, eta = X[:, j["DIABAGE2"]], X[:, j["_AGE80"]]
        X[(d < 0) | (d > eta), j["DIABAGE2"]] = np.nan
    return X


def regole_fisse_gruppo4(X, cols):
    """Tolte le 4 somme esatte; valori fisicamente impossibili -> NaN."""
    X, cols = drop_columns(X, cols, TOTALI)
    return valori_impossibili_a_nan(X, cols), cols


def fit_continue(X, cols, usa_log=None):
    """
    Impara dal TRAIN (e solo da lui):
      - i gruppi di colonne continue che mancano esattamente nelle stesse righe
        (un indicatore per gruppo: se due colonne hanno lo stesso pattern di NaN
         i loro indicatori sarebbero colonne identiche -> collinearita' esatta);
      - la mediana di ogni colonna (per i NaN);
      - quali colonne mettere in log(1+x);
      - media e deviazione standard (calcolate DOPO l'imputazione).
    """
    usa_log = USA_LOG if usa_log is None else usa_log
    idx = np.array([j for j, c in enumerate(cols) if c in CONTINUE], dtype=int)
    log_mask = np.array([usa_log and cols[j] in LOG_COLS for j in idx], dtype=bool)
    Xc = X[:, idx].astype(np.float64)
    mancanti = np.isnan(Xc)

    # 3a. un indicatore per ogni gruppo di colonne con NaN esattamente nelle stesse righe
    con_nan = np.where(mancanti.any(axis=0))[0]
    gruppi_pos = []
    if len(con_nan) > 0:
        _, primo, gruppo = np.unique(mancanti[:, con_nan], axis=1,
                                     return_index=True, return_inverse=True)
        gruppo = np.ravel(gruppo)
        for g in np.argsort(primo):                 # gruppi in ordine di prima colonna
            gruppi_pos.append(con_nan[gruppo == g])
    membri = [[cols[idx[p]] for p in pos] for pos in gruppi_pos]

    # 3b. mediana del train al posto dei NaN
    med = np.nanmedian(Xc, axis=0)
    Xc = np.where(mancanti, med, Xc)

    # 4. logaritmo per le colonne con coda lunga
    assert (Xc[:, log_mask] >= 0).all(), "valori negativi in una colonna da mettere in log"
    Xc[:, log_mask] = np.log1p(Xc[:, log_mask])

    # 5. media e deviazione standard
    mu, sd = Xc.mean(axis=0), Xc.std(axis=0)
    sd[sd == 0] = 1

    nomi = ([("log " if l else "") + cols[j] for j, l in zip(idx, log_mask)]
            + [f"NaN[{m[0]}" + (f" +{len(m)-1}]" if len(m) > 1 else "]") for m in membri])
    return {"idx": idx, "log_mask": log_mask, "gruppi_pos": gruppi_pos, "membri": membri,
            "med": med, "mu": mu, "sd": sd, "nomi": nomi}


def transform_continue(X, pk, dtype=np.float32):
    """
    Applica quanto imparato sul train a train / validation / test.
    L'indicatore di un gruppo vale 1 se ALMENO UNA colonna del gruppo e' NaN
    (calcolato PRIMA dell'imputazione).
    """
    Xc = X[:, pk["idx"]].astype(np.float64)
    miss = [np.isnan(Xc[:, pos]).any(axis=1) for pos in pk["gruppi_pos"]]
    Xc = np.where(np.isnan(Xc), pk["med"], Xc)
    Xc[:, pk["log_mask"]] = np.log1p(np.maximum(Xc[:, pk["log_mask"]], 0))
    Xc = (Xc - pk["mu"]) / pk["sd"]
    return np.hstack([Xc] + [m[:, None] for m in miss]).astype(dtype)


# ---------------------------------------------------------------------------
# One-hot delle categoriche, con il NaN come categoria a se'
# ---------------------------------------------------------------------------
def fit_categoriche(X, cols, continue_cols=None):
    """
    Impara dal TRAIN: quali colonne sono categoriche, i loro livelli e il
    riferimento (la risposta piu' frequente, che viene tolta per non creare
    colonne che sommano a 1 come il bias).
    ⚠️ SCELTA PROVVISORIA: il NaN come categoria va confrontato in CV con
    "riempire con la moda".
    """
    continue_cols = CONTINUE if continue_cols is None else continue_cols
    idx = np.array([j for j, c in enumerate(cols) if c not in continue_cols], dtype=int)
    assert not (X[:, idx] == CODICE_NAN).any(), "il valore -1 esiste gia' nei dati: cambia CODICE_NAN"
    Xk = np.nan_to_num(X[:, idx], nan=CODICE_NAN)
    livelli, riferimento = [], []
    for c in Xk.T:
        v, n = np.unique(c, return_counts=True)
        riferimento.append(v[np.argmax(n)])
        livelli.append(np.delete(v, np.argmax(n)))   # tutti tranne il riferimento
    lab = lambda v: "NaN" if v == CODICE_NAN else f"{v:g}"
    nomi = [f"{cols[j]}={lab(v)}" for j, lev in zip(idx, livelli) for v in lev]
    return {"idx": idx, "livelli": livelli, "riferimento": riferimento, "nomi": nomi}


def transform_categoriche(X, pc, dtype=np.float32):
    """Applica il one-hot imparato sul train a qualsiasi matrice (train, validation, test)."""
    Xk = np.nan_to_num(X[:, pc["idx"]], nan=CODICE_NAN)
    return np.hstack([(Xk[:, [j]] == lev[None, :]).astype(dtype) for j, lev in enumerate(pc["livelli"])])


# ---------------------------------------------------------------------------
# Collinearita' esatte: correzioni D (INSULIN_CAT), E (SMOKE100), F (artrite), G (ASTHMA3)
# ---------------------------------------------------------------------------
def correggi_collinearita(X, cols, verbose=False):
    """
    Applica (solo dove il controllo passa) le 4 correzioni delle collinearita' esatte:
      D. INSULIN_CAT: 0, 1, 2 -> 0 ("non risulta che usi insulina")
      E. SMOKE100 tolta (l'informazione e' gia' in _SMOKER3)
      F. artrite: "non ha l'artrite" -> "non lo limita"
      G. ASTHMA3 tolta (_ASTHMS1 contiene tutto)
    Ritorna (X, cols, ok).
    """
    X = X.copy()
    j = _idx(cols)
    ok = {}
    uguali = lambda a, b: bool(np.array_equal(a, b))

    if "INSULIN_CAT" in j and len(np.unique(X[:, j["INSULIN_CAT"]])) == 5:
        ins, dia = X[:, j["INSULIN_CAT"]], X[:, j["DIABETE3"]]
        c1 = uguali(ins == 2, np.isnan(dia))
        c2 = uguali(np.isin(ins, [1, 3, 4]), dia == 1)
        ok["D"] = c1 and c2
        if ok["D"]:
            X[np.isin(X[:, j["INSULIN_CAT"]], [0, 1, 2]), j["INSULIN_CAT"]] = 0
    else:
        ok["D"] = False

    ok["E"] = "SMOKE100" in j
    ok["G"] = False
    if "ASTHMA3" in j and "_ASTHMS1" in j:
        ok["G"] = uguali(X[:, j["_ASTHMS1"]] == 3, X[:, j["ASTHMA3"]] == 2)

    for c, codice in NO_ARTRITE.items():
        if c in j and "HAVARTH3" in j:
            if uguali(X[:, j[c]] == codice, X[:, j["HAVARTH3"]] == 2):
                X[X[:, j[c]] == codice, j[c]] = codice - 1     # "non ha l'artrite" -> "non lo limita"

    da_togliere = (["SMOKE100"] if ok["E"] else []) + (["ASTHMA3"] if ok["G"] else [])
    X, cols = drop_columns(X, cols, da_togliere)
    if verbose:
        print(f"   correzioni: {ok}   tolte: {da_togliere}")
    return X, cols, ok


# ---------------------------------------------------------------------------
# Matrice finale: bias + continue + categoriche
# ---------------------------------------------------------------------------
PRIORITA = {"bias": 0, "cont": 1, "oh": 2, "ind": 3}    # a parita' di colonna, chi si tiene


def _blocchi(X, pk, pc, dtype=np.float32):
    return np.hstack([np.ones((X.shape[0], 1), dtype=dtype),
                      transform_continue(X, pk, dtype),
                      transform_categoriche(X, pc, dtype)])


def fit_prep(X, cols, verbose=False):
    """
    Impara dal TRAIN tutto il preprocessing e WHICH columns tenere:
      - tolte le colonne COSTANTI (tranne il bias);
      - tolte le colonne IDENTICHE riga per riga (una per gruppo), tenendo
        quella con priorita' bias > continue > one-hot > indicatore.
    Ritorna il dizionario di parametri da passare a transform_prep.
    """
    pk = fit_continue(X, cols)
    pc = fit_categoriche(X, cols)
    nomi = np.array(["bias"] + pk["nomi"] + pc["nomi"])
    tipo = np.array(["bias"] + ["cont"] * len(pk["idx"]) + ["ind"] * len(pk["gruppi_pos"])
                    + ["oh"] * len(pc["nomi"]))
    Z = _blocchi(X, pk, pc)

    # 1. colonne costanti (tranne il bias)
    costante = Z.max(axis=0) == Z.min(axis=0)
    costante[0] = False

    # 2. colonne identiche: impronta = proiezione su un vettore casuale (a blocchi)
    r = np.random.default_rng(0).standard_normal(Z.shape[0])
    impronta = np.zeros(Z.shape[1])
    for s in range(0, Z.shape[0], 50_000):
        impronta += r[s:s + 50_000] @ Z[s:s + 50_000].astype(np.float64)
    gruppi = {}
    for k in np.where(~costante)[0]:
        gruppi.setdefault(impronta[k], []).append(k)

    togli, copie = set(np.where(costante)[0]), []
    for ks in gruppi.values():
        if len(ks) < 2:
            continue
        ks = sorted(ks, key=lambda k: (PRIORITA[tipo[k]], k))
        tenuta = ks[0]
        for k in ks[1:]:
            if np.array_equal(Z[:, k], Z[:, tenuta]):
                togli.add(k)
                copie.append((nomi[k], nomi[tenuta]))

    keep = np.array([k for k in range(Z.shape[1]) if k not in togli])
    if verbose:
        print(f"   colonne costanti tolte: {int(costante.sum())}", list(nomi[costante]))
        print(f"   colonne identiche tolte: {len(copie)}")
        for tolta, tenuta in copie:
            print(f"      {tolta:22s} = {tenuta}  (tenuta)")
    return {"pk": pk, "pc": pc, "keep": keep, "nomi": list(nomi[keep])}


def transform_prep(X, p, dtype=np.float32):
    """Applica a train / validation / test il preprocessing imparato sul train."""
    return _blocchi(X, p["pk"], p["pc"], dtype)[:, p["keep"]]


# ---------------------------------------------------------------------------
# Orchestratore: tutte le "regole fisse" in fila, una volta sola
# ---------------------------------------------------------------------------
def pulizia_preprocessing(x, cols, verbose=True):
    """
    Applica in ordine TUTTE le regole fisse della parte 2 (nessuna statistica:
    quindi il risultato e' identico su train e test e puo' essere applicato una
    volta sola, prima della cross-validation).

    Ritorna (X_pulito, cols_pulito, report) con report = dizionario con il
    numero di colonne dopo ogni passo.
    """
    report = {}

    def passo(nome, X, cols):
        report[nome] = X.shape[1]
        if verbose:
            print(f"   {nome:32s} -> {X.shape[1]} colonne")
        return X, cols

    if verbose:
        print("GRUPPO 1a - colonne grezze con codifica mista")
    x, cols, rep = rimuovi_grezze(x, cols, verbose=verbose)
    report["1a_grezze_tolte"] = [r["grezza"] for r in rep if r.get("esito") == "TOLTA"]
    passo("dopo 1a", x, cols)

    if verbose:
        print("GRUPPO 1b - IMFVPLAC -> IMFV_LAVORO; tolte EXRACT11/21")
    x, cols = trasforma_imfv(x, cols)
    passo("dopo 1b", x, cols)

    if verbose:
        print("GRUPPO 1c - _HCVU651")
    x, cols, _ = rimuovi_hcvu651(x, cols, verbose=verbose)
    passo("dopo 1c", x, cols)

    if verbose:
        print("GRUPPO 2 - NaN strutturali -> 'no'")
    ok = controlli_gruppo2(x, cols, verbose=verbose)
    x, cols = trasforma_gruppo2(x, cols, ok)
    report["gruppo2_controlli"] = ok
    passo("dopo 2", x, cols)

    if verbose:
        print("GRUPPO 3 - colonne doppie")
    x, cols = rimuovi_gruppo3(x, cols)
    passo("dopo 3", x, cols)

    if verbose:
        print("GRUPPO 4 - somme esatte tolte, valori impossibili -> NaN")
    x, cols = regole_fisse_gruppo4(x, cols)
    passo("dopo 4", x, cols)

    if verbose:
        print("GRUPPO 5 - collinearita' esatte (D, E, F, G)")
    x, cols, okc = correggi_collinearita(x, cols, verbose=verbose)
    report["correzioni_collinearita"] = okc
    passo("dopo 5", x, cols)

    report["shape_finale"] = x.shape
    return x, cols, report
