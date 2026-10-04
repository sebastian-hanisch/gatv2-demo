"""Unabhängige Orakel für GCN, GAT und GATv2 auf gerichteten Kantenlisten: Schleifen-Referenz (Vorwärtsrechnung, Aufmerksamkeit je Kante), Äquivarianz, Verlust und Gradienten aller Parameter gegen zentrale Differenzen
auf Zufallsgraphen (auch ohne Kanten), Aufbau der Lager-Suche, Trefferquote/Matrix/Rangfolge-Invarianz der Aufmerksamkeit per eigener Schleife."""

import numpy as np
import pytest

import g2_algorithm as A
import g2_evaluation as E
import g2_scenario as S


def _rand_dir(rng, n, p):
    M = (rng.random((n, n)) < p).astype(float)
    np.fill_diagonal(M, 0.0)
    return M


def _lk(z):
    return z if z > 0 else 0.2 * z


def _senders(Ad, i):
    return [i] + [j for j in range(len(Ad)) if Ad[i, j] > 0]


def _gat_loop(H, W, a_s, a_d, Ad):
    n = len(Ad)
    heads, f = a_s.shape
    Wh = H @ W
    out = np.zeros((n, heads * f))
    for h in range(heads):
        Wf = Wh[:, h * f:(h + 1) * f]
        for i in range(n):
            nb = _senders(Ad, i)
            e = np.array([_lk(a_s[h] @ Wf[i] + a_d[h] @ Wf[j]) for j in nb])
            al = np.exp(e - e.max())
            al /= al.sum()
            out[i, h * f:(h + 1) * f] = sum(al[q] * Wf[j] for q, j in enumerate(nb))
    return out


def _v2_loop(H, Wl, Wr, a, Ad):
    n = len(Ad)
    heads, f = a.shape
    L, R = H @ Wl, H @ Wr
    out = np.zeros((n, heads * f))
    for h in range(heads):
        Lf, Rf = L[:, h * f:(h + 1) * f], R[:, h * f:(h + 1) * f]
        for i in range(n):
            nb = _senders(Ad, i)
            e = np.array([a[h] @ np.array([_lk(v) for v in Lf[i] + Rf[j]]) for j in nb])
            al = np.exp(e - e.max())
            al /= al.sum()
            out[i, h * f:(h + 1) * f] = sum(al[q] * Rf[j] for q, j in enumerate(nb))
    return out


def _loop_logits(kind, p, X, Ad, mode):
    relu = lambda v: np.maximum(v, 0)
    if kind == "gat":
        return _gat_loop(relu(_gat_loop(X, p["W1"], p["as1"], p["ad1"], Ad)), p["W2"], p["as2"], p["ad2"], Ad)
    if kind == "gatv2":
        return _v2_loop(relu(_v2_loop(X, p["Wl1"], p["Wr1"], p["a1"], Ad)), p["Wl2"], p["Wr2"], p["a2"], Ad)
    n = len(Ad)
    din = [len(_senders(Ad, i)) for i in range(n)]

    def wgt(i, j):
        return {"self": float(i == j), "mean": 1.0 / din[i], "sym": 1.0 / np.sqrt(din[i] * din[j])}[mode]

    def prop(H):
        return np.array([sum(wgt(i, j) * H[j] for j in _senders(Ad, i)) for i in range(n)])
    return prop(relu(prop(X) @ p["W1"])) @ p["W2"]


@pytest.mark.parametrize("kind,mode", [("gcn", "mean"), ("gcn", "sym"), ("mlp", "self"), ("gat", None), ("gatv2", None)])
def test_forward_loss_and_gradients_match_independent_references(kind, mode):
    rng = np.random.default_rng(6)
    for t in range(12):
        n = int(rng.integers(1, 9))
        Ad = _rand_dir(rng, n, rng.random())
        if mode == "sym":
            Ad = np.maximum(Ad, Ad.T)                                     # 'sym' ist für ungerichtete Graphen gedacht
        d, heads, f, K = int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(2, 4))
        kk = "gcn" if kind == "mlp" else kind
        p = A.init_params(kk, d, heads, f, K, t, hidden=int(rng.integers(2, 6)))
        for k in p:
            p[k] = p[k] + rng.normal(size=p[k].shape) * 0.3
        X = rng.normal(size=(n, d))
        g = A.edges_of(Ad)
        w = A.gcn_weights(g, mode) if mode else None
        lg = A.forward(kk, p, X, g, w)[0]
        assert np.allclose(lg, _loop_logits(kind if kind != "mlp" else "gcn", p, X, Ad, mode))
        perm = rng.permutation(n)
        gp = A.edges_of(Ad[np.ix_(perm, perm)])
        assert np.allclose(A.forward(kk, p, X[perm], gp, A.gcn_weights(gp, mode) if mode else None)[0], lg[perm])
        y = rng.integers(0, K, size=n)
        tr = rng.random(n) < 0.6
        tr[0] = True
        wd = float(rng.choice([0.0, 5e-4, 0.05]))
        loss, gr = A.loss_and_grads(kk, p, X, y, tr, g, wd, w)
        ce = np.mean([np.log(np.exp(lg[i] - lg[i].max()).sum()) + lg[i].max() - lg[i][y[i]] for i in np.flatnonzero(tr)])
        assert loss == pytest.approx(ce + 0.5 * wd * sum((p[k] ** 2).sum() for k in p if k.startswith("W")), abs=1e-7)
        for k in p:
            idx = tuple(rng.integers(0, s) for s in p[k].shape)
            pp = {a: b.copy() for a, b in p.items()}
            pm = {a: b.copy() for a, b in p.items()}
            pp[k][idx] += 1e-6
            pm[k][idx] -= 1e-6
            num = (A.loss_and_grads(kk, pp, X, y, tr, g, wd, w)[0] - A.loss_and_grads(kk, pm, X, y, tr, g, wd, w)[0]) / 2e-6
            assert gr[k][idx] == pytest.approx(num, rel=1e-4, abs=1e-5), k


@pytest.mark.parametrize("kind", ["gat", "gatv2"])
def test_lookup_construction_and_attention_metrics_match_an_independent_loop(kind):
    rng = np.random.default_rng(7)
    for t in range(6):
        K, nc = int(rng.integers(2, 6)), int(rng.integers(3, 9))
        lk = S.generate_lookup(K, nc, t)
        for c in range(nc):                                               # Etikett des Kunden = Zone des Lagers mit passendem Schlüssel
            base = lk.cust[c]
            keys = [int(np.argmax(lk.X[base + 1 + d, K:2 * K])) for d in range(K)]
            zones = [int(np.argmax(lk.X[base + 1 + d, 2 * K:])) for d in range(K)]
            assert sorted(keys) == list(range(K)) and lk.y[base] == zones[keys.index(int(np.argmax(lk.X[base, :K])))]
            assert set(np.flatnonzero(lk.A[base])) == set(range(base + 1, base + 1 + K)) and lk.A[:, base].sum() == 0
        tr, te = S.lookup_masks(lk, 0.5, t)
        p = A.init_params(kind, 2 * K + 3, 2, 3, 3, t)
        model = type("M", (), {"kind": kind, "params": p})()
        alpha, g = A.attention_layer1(model, lk.A, lk.X)

        def depot_att(X_):
            res = np.zeros((nc, K, 2))
            for c in range(nc):
                base = lk.cust[c]
                nb = _senders(lk.A, base)
                for h in range(2):
                    if kind == "gat":
                        Wf = (X_ @ p["W1"])[:, h * 3:(h + 1) * 3]
                        e = np.array([_lk(p["as1"][h] @ Wf[base] + p["ad1"][h] @ Wf[j]) for j in nb])
                    else:
                        Lf, Rf = (X_ @ p["Wl1"])[:, h * 3:(h + 1) * 3], (X_ @ p["Wr1"])[:, h * 3:(h + 1) * 3]
                        e = np.array([p["a1"][h] @ np.array([_lk(v) for v in Lf[base] + Rf[j]]) for j in nb])
                    al = np.exp(e - e.max())
                    res[c, :, h] = (al / al.sum())[1:]
            return res
        ref = depot_att(lk.X)
        assert np.allclose(E.depot_attention(alpha, g, lk), ref)
        att = ref.mean(axis=2)
        att = att / att.sum(axis=1, keepdims=True)
        te_c = np.isin(lk.cust, np.flatnonzero(te))
        hits = np.array([np.argmax(att[c]) == list(lk.keys[c]).index(lk.query[c]) for c in range(nc)])
        hit, M = E.attention_metrics(alpha, g, lk, te_c)
        if te_c.any():
            assert hit == pytest.approx(float(hits[te_c].mean()))
        Mref, cc = np.zeros((K, K)), np.zeros(K)
        for c in range(nc):
            for slot in range(K):
                Mref[lk.query[c], lk.keys[c][slot]] += att[c][slot]
            cc[lk.query[c]] += 1
        assert np.allclose(M, Mref / np.maximum(cc, 1)[:, None])
        base_order = np.argsort(ref, axis=1)
        same = total = 0
        for q in range(K):
            o2 = np.argsort(depot_att(S.with_query(lk, np.full(nc, q))), axis=1)
            for c in range(nc):
                if lk.query[c] != q:
                    same += int(np.all(o2[c] == base_order[c], axis=0).sum())
                    total += 2
        assert E.rank_invariance(model, lk) == pytest.approx(same / total)
        if kind == "gat":
            assert E.rank_invariance(model, lk) == pytest.approx(1.0)
