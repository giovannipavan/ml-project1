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
