"""Zwei Vehikel.

**E "Lager-Suche":** jeder Kunde hat seinen eigenen Stern: der Kunde und K Lager. Der Kunde kennt einen Artikelcode q (Anfrage-Schlüssel), jedes seiner Lager führt genau einen Code (Schlüssel = Permutation von 0..K-1) und liegt in
einer Servicezone (Wert). Gesucht ist die Zone des Lagers, dessen Code zum Artikelcode passt. Die Kanten sind **gerichtet**: die Lager senden an den Kunden, nicht umgekehrt - ein Lager weiß also nicht, wonach der Kunde sucht.
Merkmale: Anfrage-Schlüssel (one-hot, nur beim Kunden) | Schlüssel (one-hot, nur beim Lager) | Zone (one-hot, nur beim Lager). Weil Schlüssel und Zonen je Kunde neu gemischt werden, lässt sich nichts auswendig lernen.

**D "Liefergebiete"** (wie gcn-demo): Kunden in räumlichen Gebietstypen, vier verrauschte Merkmale, Graph der nächsten Nachbarn; ein Anteil der Kanten kann durch zufällige ersetzt werden."""

from dataclasses import dataclass

import numpy as np

import g2_constants as C


@dataclass(frozen=True)
class Lookup:
    X: np.ndarray            # (n, 2K + Klassen)
    A: np.ndarray            # (n, n) gerichtet: A[i, j] = 1, wenn j an i sendet (ohne Schleifen)
    y: np.ndarray            # (n,) Zone (bei Kunden: die gesuchte; bei Lagern: die eigene)
    cust: np.ndarray         # Knotenindizes der Kunden
    query: np.ndarray        # Anfrage-Schlüssel je Kunde
    keys: np.ndarray         # (n_cust, K) Schlüssel der Lager des Kunden (Lager d hat Knotenindex cust + 1 + d)
    K: int
    seed: int

    @property
    def n_cust(self):
        return len(self.cust)


def generate_lookup(K=C.DEFAULT_K, n_cust=C.DEFAULT_CUST, seed=0):
    rng = np.random.default_rng(seed)
    n = n_cust * (K + 1)
    X = np.zeros((n, 2 * K + C.N_CLASSES))
    A = np.zeros((n, n))
    y = np.zeros(n, dtype=int)
    cust = np.arange(n_cust) * (K + 1)
    query = rng.integers(0, K, n_cust)
    keys = np.array([rng.permutation(K) for _ in range(n_cust)])
    vals = rng.integers(0, C.N_CLASSES, (n_cust, K))
    for c in range(n_cust):
        base = cust[c]
        X[base, query[c]] = 1.0
        for d in range(K):
            X[base + 1 + d, K + keys[c, d]] = 1.0
            X[base + 1 + d, 2 * K + vals[c, d]] = 1.0
            A[base, base + 1 + d] = 1.0
            y[base + 1 + d] = vals[c, d]
        y[base] = vals[c, int(np.flatnonzero(keys[c] == query[c])[0])]
    return Lookup(X, A, y, cust, query, keys, K, int(seed))


def lookup_masks(lk, known, seed):
    """Trainingsknoten = zufällig `known` (Anteil) der Kunden; Prüfknoten = die übrigen Kunden."""
    rng = np.random.default_rng([seed, 1])
    m = max(1, int(round(known * lk.n_cust)))
    tr = np.zeros(len(lk.y), dtype=bool)
    tr[lk.cust[rng.permutation(lk.n_cust)[:m]]] = True
    te = np.zeros(len(lk.y), dtype=bool)
    te[lk.cust] = True
    return tr, te & ~tr


def with_query(lk, new_query):
    """Dieselben Lager, aber andere Anfrage-Schlüssel (Merkmale der Kunden ersetzt); Rückgabe X."""
    X = lk.X.copy()
    K = lk.K
    X[lk.cust, :K] = 0.0
    X[lk.cust, new_query] = 1.0
    return X


# --- Vehikel D ------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Areas:
    xy: np.ndarray
    X: np.ndarray
    y: np.ndarray
    A: np.ndarray
    n_classes: int
    seed: int

    @property
    def n(self):
        return len(self.y)

    def edge_homophily(self):
        i, j = np.nonzero(np.triu(self.A, 1))
        return float(np.mean(self.y[i] == self.y[j])) if len(i) else 1.0


def _centers(rng, k):
    for _ in range(1000):
        c = rng.random((k, 2)) * C.AREA
        d = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(k) * 1e9
        if d.min() >= 0.28 * C.AREA:
            return c
    return c


def knn_adjacency(xy, k):
    n = len(xy)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(n) * 1e9
    idx = np.argsort(d, axis=1)[:, :k]
    A = np.zeros((n, n))
    A[np.repeat(np.arange(n), k), idx.ravel()] = 1.0
    return np.maximum(A, A.T)


def rewire(A, fraction, rng):
    """Ersetzt einen Anteil der Kanten durch gleich viele zufällige (keine Schleifen, keine Doppelten)."""
    if fraction <= 0:
        return A.copy()
    n = len(A)
    iu = np.array(np.nonzero(np.triu(A, 1))).T
    m = len(iu)
    n_swap = int(round(fraction * m))
    keep = np.ones(m, dtype=bool)
    keep[rng.permutation(m)[:n_swap]] = False
    B = np.zeros_like(A)
    B[iu[keep, 0], iu[keep, 1]] = 1.0
    B = np.maximum(B, B.T)
    added = 0
    while added < n_swap:
        i, j = rng.integers(0, n, 2)
        if i != j and B[i, j] == 0:
            B[i, j] = B[j, i] = 1.0
            added += 1
    return B


def generate_areas(n=C.AREA_N, n_classes=3, k=C.AREA_NEIGHBORS, noise=C.AREA_NOISE, wrong=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * C.AREA
    centers = _centers(rng, n_classes)
    y = np.argmin(np.linalg.norm(xy[:, None] - centers[None], axis=2), axis=1)
    means = rng.normal(size=(n_classes, 4))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    X = means[y] + noise * rng.normal(size=(n, 4)) * 0.5
    A = rewire(knn_adjacency(xy, k), wrong, np.random.default_rng([seed, 99]))
    return Areas(xy, X, y, A, n_classes, int(seed))


def split_areas(y, per_class, seed):
    rng = np.random.default_rng([seed, 7])
    train = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        train[rng.permutation(idx)[:min(per_class, len(idx))]] = True
    return train
