"""Auswertung: GCN, GAT und GATv2 auf der Lager-Suche (Analyse, Aufmerksamkeit, Interventionstest) und drei Experimente (Zahl der Lager, Zahl der Köpfe, falsche Kanten im Liefergebiet)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import g2_algorithm as A
import g2_constants as C
import g2_scenario as S


@dataclass(frozen=True)
class Settings:
    K: int = C.DEFAULT_K
    n_cust: int = C.DEFAULT_CUST
    known: float = C.DEFAULT_KNOWN
    heads: int = C.DEFAULT_HEADS
    dim: int = C.DEFAULT_DIM
    seed: int = 3


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def depot_edges(g, lk):
    """Indizes der Kanten Lager -> Kunde (ohne Schleifen) in der Reihenfolge Kunde, Lager 0..K-1: Array (n_cust, K)."""
    n_cust, K = lk.n_cust, lk.K
    # Kanten sind nach Empfänger sortiert; der Kunde c hat die Eingangskanten von c (Schleife) und c+1..c+K (Lager) in aufsteigender Senderreihenfolge
    start = g.ptr_dst[lk.cust]
    return start[:, None] + np.arange(K)[None, :]                       # Schleife hat den kleinsten Sender (= c selbst), dann folgen die Lager


def depot_attention(alpha, g, lk):
    """alpha (E, Köpfe) -> Aufmerksamkeit der Kunden auf ihre K Lager: (n_cust, K, Köpfe); die Schleife des Kunden liegt VOR den Lagern (kleinster Sender), das erste Lager hat also den Kanten-Offset 1."""
    idx = depot_edges(g, lk) + 1
    return alpha[idx]


def attention_metrics(alpha, g, lk, test_customers):
    """Aus der Kopf-gemittelten Aufmerksamkeit auf die Lager (ohne Schleife, auf Summe 1 normiert): Trefferquote (höchste Aufmerksamkeit auf dem Lager mit passendem Schlüssel) und Matrix M[q, k]
    (Anfrage-Schlüssel q gegen Schlüssel k des Lagers)."""
    att = depot_attention(alpha, g, lk).mean(axis=2)                     # (n_cust, K)
    att = att / att.sum(axis=1, keepdims=True)
    match_slot = np.array([int(np.flatnonzero(lk.keys[c] == lk.query[c])[0]) for c in range(lk.n_cust)])
    hit = (att.argmax(axis=1) == match_slot)
    K = lk.K
    M = np.zeros((K, K))
    cnt = np.zeros(K)
    for c in range(lk.n_cust):
        M[lk.query[c], lk.keys[c]] += att[c]
        cnt[lk.query[c]] += 1
    M = M / np.maximum(cnt, 1)[:, None]
    return float(hit[test_customers].mean()), M


def rank_invariance(model, lk):
    """Interventionstest: Anteil der (Kunde, Kopf, andere Anfrage)-Fälle, in denen die Rangfolge der Lager nach Aufmerksamkeit (erste Schicht) dieselbe bleibt wie mit der echten Anfrage.
    Bei GAT ist das exakt 1 (die Rangfolge hängt nur vom Lager ab), bei GATv2 kann sie sich mit der Anfrage ändern."""
    K = lk.K
    alpha0, g = A.attention_layer1(model, lk.A, lk.X)
    base = np.argsort(depot_attention(alpha0, g, lk), axis=1)            # (n_cust, K, Köpfe)
    same = total = 0
    for q in range(K):
        Xq = S.with_query(lk, np.full(lk.n_cust, q))
        alpha, _ = A.attention_layer1(model, lk.A, Xq)
        order = np.argsort(depot_attention(alpha, g, lk), axis=1)
        keep = lk.query != q
        eq = np.all(order == base, axis=1)                                # (n_cust, Köpfe)
        same += int(eq[keep].sum())
        total += int(eq[keep].size)
    return same / total


@dataclass
class Analysis:
    settings: Settings
    lookup: S.Lookup
    train_mask: np.ndarray
    test_mask: np.ndarray
    models: dict
    acc: dict
    chance: float
    hit: dict               # Trefferquote der Aufmerksamkeit (gat, gatv2)
    M: dict                 # Matrix Anfrage x Schlüssel (gat, gatv2)
    invariance: dict        # Rangfolge bleibt bei anderer Anfrage gleich (gat, gatv2)
    alpha: dict             # Aufmerksamkeit je Kante (gat, gatv2)
    graph: A.Edges


@lru_cache(maxsize=24)
def analyse(settings):
    lk = S.generate_lookup(settings.K, settings.n_cust, settings.seed)
    tr, te = S.lookup_masks(lk, settings.known, settings.seed)
    models, acc = {}, {}
    for kind in ("gcn", "gat", "gatv2"):
        m = A.train(kind, lk.A, lk.X, lk.y, tr, te, seed=settings.seed, heads=settings.heads, f=settings.dim, gcn_mode="mean")
        models[kind] = m
        acc[kind] = m.history["test_acc"][-1]
    test_customers = np.isin(lk.cust, np.flatnonzero(te))
    hit, M, inv, alpha = {}, {}, {}, {}
    g = None
    for kind in ("gat", "gatv2"):
        alpha[kind], g = A.attention_layer1(models[kind], lk.A, lk.X)
        hit[kind], M[kind] = attention_metrics(alpha[kind], g, lk, test_customers)
        inv[kind] = rank_invariance(models[kind], lk)
    chance = float(np.bincount(lk.y[te]).max() / te.sum())
    return Analysis(settings, lk, tr, te, models, acc, chance, hit, M, inv, alpha, g)


# --- Experiment 1: Zahl der Lager --------------------------------------------------------------------------------------------------------------


def _lookup_run(K, n_cust, known, heads, dim, seed, kinds, epochs=C.EPOCHS):
    lk = S.generate_lookup(K, n_cust, seed)
    tr, te = S.lookup_masks(lk, known, seed)
    out = {}
    for kind in kinds:
        m = A.train(kind, lk.A, lk.X, lk.y, tr, te, seed=seed, heads=heads, f=dim, gcn_mode="mean", epochs=epochs)
        out[kind] = m.history["test_acc"][-1]
    return out, float(np.bincount(lk.y[te]).max() / te.sum())


def keys_experiment(levels=None, seeds=None, n_cust=None, known=None, heads=None, dim=None):
    levels = C.K_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS_LOOKUP if seeds is None else seeds
    n_cust = C.EXP_CUST if n_cust is None else n_cust
    known = C.DEFAULT_KNOWN if known is None else known
    heads = C.DEFAULT_HEADS if heads is None else heads
    dim = C.DEFAULT_DIM if dim is None else dim
    rows = []
    for K in levels:
        res = [_lookup_run(K, n_cust, known, heads, dim, s, ("gcn", "gat", "gatv2")) for s in seeds]
        row = {"K": K, "heads": heads, "n_seeds": len(res), "chance": float(np.mean([r[1] for r in res]))}
        for kind in ("gcn", "gat", "gatv2"):
            row[kind], row[kind + "_se"] = _mean_se([r[0][kind] for r in res])
        d = np.array([r[0]["gatv2"] - r[0]["gat"] for r in res])
        row["diff"], row["diff_se"] = _mean_se(d)
        rows.append(row)
    return rows


# --- Experiment 2: Zahl der Köpfe ---------------------------------------------------------------------------------------------------------------


def heads_experiment(levels=None, seeds=None, K=None, n_cust=None, known=None, dim=None):
    levels = C.HEAD_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS_LOOKUP if seeds is None else seeds
    K = C.DEFAULT_K if K is None else K
    n_cust = C.EXP_CUST if n_cust is None else n_cust
    known = C.DEFAULT_KNOWN if known is None else known
    dim = C.DEFAULT_DIM if dim is None else dim
    rows = []
    for h in levels:
        res = [_lookup_run(K, n_cust, known, h, dim, s, ("gat", "gatv2")) for s in seeds]
        row = {"heads": h, "K": K, "n_seeds": len(res)}
        for kind in ("gat", "gatv2"):
            row[kind], row[kind + "_se"] = _mean_se([r[0][kind] for r in res])
        d = np.array([r[0]["gatv2"] - r[0]["gat"] for r in res])
        row["diff"], row["diff_se"] = _mean_se(d)
        rows.append(row)
    return rows


# --- Experiment 3: falsche Kanten im Liefergebiet ---------------------------------------------------------------------------------------------


def wrong_experiment(levels=None, seeds=None):
    levels = C.WRONG_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS_AREA if seeds is None else seeds
    rows = []
    for w in levels:
        res = {k: [] for k in ("gcn", "mlp", "gat", "gatv2")}
        hom = []
        for s in seeds:
            g = S.generate_areas(wrong=w, seed=s)
            tr = S.split_areas(g.y, C.AREA_LABELS, s)
            te = ~tr
            hom.append(g.edge_homophily())
            for kind in res:
                m = A.train(kind, g.A, g.X, g.y, tr, te, seed=s, heads=C.AREA_HEADS, f=C.AREA_DIM, epochs=C.AREA_EPOCHS, gcn_mode="sym")
                res[kind].append(m.history["test_acc"][-1])
        row = {"wrong": w, "homophily": float(np.mean(hom)), "n_seeds": len(seeds)}
        for kind, v in res.items():
            row[kind], row[kind + "_se"] = _mean_se(v)
        d = np.array(res["gatv2"]) - np.array(res["gat"])
        row["diff_v2_gat"], row["diff_v2_gat_se"] = _mean_se(d)
        d2 = np.array(res["gatv2"]) - np.array(res["gcn"])
        row["diff_v2_gcn"], row["diff_v2_gcn_se"] = _mean_se(d2)
        row["v2_wins_vs_gcn"] = int(np.sum(d2 > 0))
        rows.append(row)
    return rows
