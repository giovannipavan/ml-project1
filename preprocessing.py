"""Preprocessing per il Project 1 (BRFSS 2015, target _MICHD).

Trasforma le colonne pulite (x_base, 95 colonne con NaN) nella matrice numerica
da dare ai modelli (~256 colonne, senza NaN):

    bias | continue (log + z-score) | indicatori di missing | one-hot delle categoriche

Regola d'oro: tutte le statistiche (mediane, medie, deviazioni standard, livelli
del one-hot, colonne da tenere) si imparano SOLO sul training con `fit_prep` e si
applicano identiche a validation e test con `transform_prep`. Nella
cross-validation `fit_prep` va chiamata dentro ogni fold.

Uso tipico:
    p, X_tr = fit_transform_prep(x_base[train], cols_base)
    X_va    = transform_prep(x_base[val], p)

Le scelte sono spiegate in CUTTING_COLONNE.md (parte 2, sezioni 15-17).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Colonne continue (quantità). Tutte le altre sono trattate come categoriche.
# Le colonne della lista che non sono presenti nei dati vengono ignorate.
CONTINUE = {
    "_AGE80", "CHILDREN", "PHYSHLTH", "MENTHLTH", "POORHLTH", "DIABAGE2",
    "WTKG3", "HTM4", "_BMI5", "MAXVO2_",
    "AVEDRNK2", "DRNK3GE5", "MAXDRNKS", "DROCDY3_", "_DRNKWEK",
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_",
    "_FRUTSUM", "_VEGESUM",
    "STRFREQ_", "PAFREQ1_", "PAFREQ2_", "PADUR1_", "PADUR2_", "PAMIN11_", "PAMIN21_",
    "PA1MIN_", "PAVIG11_", "PAVIG21_", "PA1VIGM_", "_MINAC11", "_MINAC21",
    "METVL11_", "METVL21_",
}

# Continue con coda lunga: si applica log(1 + x) prima della standardizzazione.
LOG_COLS = {
    "AVEDRNK2", "DRNK3GE5", "MAXDRNKS", "_DRNKWEK",                              # alcol
    "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_",      # frutta e verdura
    "PADUR1_", "PADUR2_", "PAFREQ1_", "PAFREQ2_", "STRFREQ_",                    # attività fisica
    "_MINAC11", "_MINAC21", "PAMIN11_", "PAMIN21_", "PAVIG11_", "PAVIG21_",
}
USA_LOG = True        # scelta da confrontare in CV (con / senza logaritmo)

CODICE_NAN = -1       # valore che sostituisce il NaN prima del one-hot

# Tra colonne identiche si tiene quella con priorità più bassa.
PRIORITA = {"bias": 0, "cont": 1, "oh": 2, "ind": 3}


# ---------------------------------------------------------------------------
# Continue: mediana + indicatore di missing, logaritmo, z-score
# ---------------------------------------------------------------------------

def fit_continue(X, cols, usa_log=None):
    """Impara dal train: gruppi di NaN, mediane, colonne in log, media e dev. std."""
    usa_log = USA_LOG if usa_log is None else usa_log
    idx = np.array([j for j, c in enumerate(cols) if c in CONTINUE], dtype=int)
    log_mask = np.array([usa_log and cols[j] in LOG_COLS for j in idx], dtype=bool)
    Xc = X[:, idx].astype(np.float64)
    mancanti = np.isnan(Xc)

    # un indicatore per ogni gruppo di colonne con NaN esattamente nelle stesse righe
    con_nan = np.where(mancanti.any(axis=0))[0]
    gruppi_pos = []
    if len(con_nan) > 0:
        _, primo, gruppo = np.unique(mancanti[:, con_nan], axis=1,
                                     return_index=True, return_inverse=True)
        gruppo = np.ravel(gruppo)
        for g in np.argsort(primo):
            gruppi_pos.append(con_nan[gruppo == g])
    membri = [[cols[idx[p]] for p in pos] for pos in gruppi_pos]

    # mediana del train al posto dei NaN
    med = np.nanmedian(Xc, axis=0)
    Xc = np.where(mancanti, med, Xc)

    # logaritmo per le colonne con coda lunga
    if log_mask.any():
        assert (Xc[:, log_mask] >= 0).all(), "valori negativi in una colonna da mettere in log"
        Xc[:, log_mask] = np.log1p(Xc[:, log_mask])

    # media e deviazione standard
    mu, sd = Xc.mean(axis=0), Xc.std(axis=0)
    sd[sd == 0] = 1

    nomi = ([("log " if l else "") + cols[j] for j, l in zip(idx, log_mask)]
            + [f"NaN[{m[0]}" + (f" +{len(m) - 1}]" if len(m) > 1 else "]") for m in membri])
    return {"idx": idx, "log_mask": log_mask, "gruppi_pos": gruppi_pos, "membri": membri,
            "med": med, "mu": mu, "sd": sd, "nomi": nomi}


def transform_continue(X, pk, dtype=np.float32):
    """Applica quanto imparato sul train.
    Indicatore di un gruppo = 1 se ALMENO UNA colonna del gruppo è NaN."""
    Xc = X[:, pk["idx"]].astype(np.float64)
    miss = [np.isnan(Xc[:, pos]).any(axis=1) for pos in pk["gruppi_pos"]]
    Xc = np.where(np.isnan(Xc), pk["med"], Xc)
    if pk["log_mask"].any():
        Xc[:, pk["log_mask"]] = np.log1p(np.maximum(Xc[:, pk["log_mask"]], 0))
    Xc = (Xc - pk["mu"]) / pk["sd"]
    return np.hstack([Xc] + [m[:, None] for m in miss]).astype(dtype)


# ---------------------------------------------------------------------------
# Categoriche: one-hot con NaN come categoria a sé (scelta provvisoria)
# ---------------------------------------------------------------------------

def fit_categoriche(X, cols):
    """Impara dal train i livelli di ogni categorica; il più frequente fa da riferimento."""
    idx = np.array([j for j, c in enumerate(cols) if c not in CONTINUE], dtype=int)
    assert not (X[:, idx] == CODICE_NAN).any(), "il valore CODICE_NAN esiste già nei dati"
    Xk = np.nan_to_num(X[:, idx], nan=CODICE_NAN)
    livelli, riferimento = [], []
    for c in Xk.T:
        v, n = np.unique(c, return_counts=True)
        riferimento.append(v[np.argmax(n)])
        livelli.append(np.delete(v, np.argmax(n)))          # tutti tranne il riferimento
    lab = lambda v: "NaN" if v == CODICE_NAN else f"{v:g}"
    nomi = [f"{cols[j]}={lab(v)}" for j, lev in zip(idx, livelli) for v in lev]
    return {"idx": idx, "livelli": livelli, "riferimento": riferimento, "nomi": nomi}


def transform_categoriche(X, pc, dtype=np.float32):
    """One-hot con i livelli imparati sul train (risposta mai vista -> riga di zeri)."""
    Xk = np.nan_to_num(X[:, pc["idx"]], nan=CODICE_NAN)
    blocchi = [(Xk[:, [j]] == lev[None, :]).astype(dtype) for j, lev in enumerate(pc["livelli"])]
    if not blocchi:
        return np.zeros((X.shape[0], 0), dtype=dtype)
    return np.hstack(blocchi)


# ---------------------------------------------------------------------------
# Matrice finale: bias + continue + categoriche, senza colonne identiche o costanti
# ---------------------------------------------------------------------------

def _blocchi(X, pk, pc, dtype=np.float32):
    return np.hstack([np.ones((X.shape[0], 1), dtype=dtype),
                      transform_continue(X, pk, dtype),
                      transform_categoriche(X, pc, dtype)])


def fit_transform_prep(X, cols, usa_log=None, verbose=False):
    """Impara il preprocessing dal train e restituisce (p, matrice del train)."""
    pk = fit_continue(X, cols, usa_log)
    pc = fit_categoriche(X, cols)
    nomi = np.array(["bias"] + pk["nomi"] + pc["nomi"])
    tipo = np.array(["bias"] + ["cont"] * len(pk["idx"]) + ["ind"] * len(pk["gruppi_pos"])
                    + ["oh"] * len(pc["nomi"]))
    Z = _blocchi(X, pk, pc)

    # 1. colonne costanti (tranne il bias)
    costante = Z.max(axis=0) == Z.min(axis=0)
    costante[0] = False

    # 2. colonne identiche: impronta = proiezione su un vettore casuale, poi verifica esatta
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

    keep = np.array([k for k in range(Z.shape[1]) if k not in togli], dtype=int)
    if verbose:
        print(f"colonne costanti tolte: {int(costante.sum())}", list(nomi[costante]))
        print(f"colonne identiche tolte: {len(copie)}")
        for tolta, tenuta in copie:
            print(f"   {tolta:22s} = {tenuta}  (tenuta)")
    p = {"pk": pk, "pc": pc, "keep": keep,
         "nomi": [str(s) for s in nomi[keep]], "tipo": [str(s) for s in tipo[keep]]}
    return p, Z[:, keep]


def fit_prep(X, cols, usa_log=None, verbose=False):
    """Impara dal train tutto il preprocessing e quali colonne tenere."""
    return fit_transform_prep(X, cols, usa_log, verbose)[0]


def transform_prep(X, p, dtype=np.float32):
    """Applica a train / validation / test il preprocessing imparato sul train."""
    return _blocchi(X, p["pk"], p["pc"], dtype)[:, p["keep"]]
