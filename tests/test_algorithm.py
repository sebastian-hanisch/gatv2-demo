"""Kern auf Kantenlisten: Struktur, GAT und GATv2 gegen dichte Vergleichsrechnungen, Handrechnungen (Rangfolge kippt nur bei GATv2), Gradienten gegen zentrale Differenzen, GCN-Gewichte, Training."""

import numpy as np
import pytest

import g2_algorithm as A
import g2_scenario as S


def dense_gat(H, W, a_s, a_d, mask):
    """Dichte Vergleichsrechnung GAT: alpha[i, j] für j im Eingang von i."""
    n = H.shape[0]
    heads, f = a_s.shape
    Wh = (H @ W).reshape(n, heads, f)
    out = np.zeros((n, heads, f))
    for k in range(heads):
        for i in range(n):
            nb = [j for j in range(n) if mask[i, j]]
            e = np.array([Wh[i, k] @ a_s[k] + Wh[j, k] @ a_d[k] for j in nb])
            e = np.where(e > 0, e, 0.2 * e)
            al = np.exp(e - e.max())
            al /= al.sum()
            out[i, k] = sum(al[q] * Wh[j, k] for q, j in enumerate(nb))
    return out.reshape(n, heads * f)


def dense_gatv2(H, Wl, Wr, a, mask):
    n = H.shape[0]
    heads, f = a.shape
    L = (H @ Wl).reshape(n, heads, f)
    R = (H @ Wr).reshape(n, heads, f)
    out = np.zeros((n, heads, f))
    for k in range(heads):
        for i in range(n):
            nb = [j for j in range(n) if mask[i, j]]
            u = np.array([L[i, k] + R[j, k] for j in nb])
            e = np.where(u > 0, u, 0.2 * u) @ a[k]
            al = np.exp(e - e.max())
            al /= al.sum()
            out[i, k] = sum(al[q] * R[j, k] for q, j in enumerate(nb))
    return out.reshape(n, heads * f)


def random_graph(n, p, seed, directed):
    rng = np.random.default_rng(seed)
    Am = (rng.random((n, n)) < p).astype(float)
    np.fill_diagonal(Am, 0)
    return Am if directed else np.maximum(Am, Am.T)


# --- Struktur --------------------------------------------------------------------------------------------------------------------------------


def test_edges_are_sorted_by_receiver_with_self_loops_and_segment_sums_match_bincount():
    Am = random_graph(15, 0.2, 1, directed=True)
    g = A.edges_of(Am)
    assert np.all(np.diff(g.dst) >= 0) and g.n_edges == int(Am.sum()) + 15
    assert all(((g.dst == i) & (g.src == i)).sum() == 1 for i in range(15))
    v = np.random.default_rng(2).normal(size=(g.n_edges, 3))
    for i in range(15):
        assert g.seg_dst(v)[i] == pytest.approx(v[g.dst == i].sum(axis=0)) and g.seg_src(v)[i] == pytest.approx(v[g.src == i].sum(axis=0))


def test_edge_softmax_sums_to_one_per_receiver():
    g = A.edges_of(random_graph(12, 0.3, 3, True))
    e = np.random.default_rng(1).normal(size=(g.n_edges, 2)) * 5
    al = A.edge_softmax(e, g)
    assert g.seg_dst(al) == pytest.approx(np.ones((12, 2))) and np.all(al > 0)


def test_gcn_weights_match_the_dense_normalized_adjacency_and_row_means():
    Am = random_graph(14, 0.25, 4, directed=False)
    g = A.edges_of(Am)
    W = np.zeros((14, 14))
    W[g.dst, g.src] = A.gcn_weights(g, "sym")
    B = Am + np.eye(14)
    d = B.sum(axis=1)
    assert W == pytest.approx(B / np.sqrt(d[:, None] * d[None, :]))
    wm = A.gcn_weights(g, "mean")
    assert g.seg_dst(wm) == pytest.approx(np.ones(14)) and A.gcn_weights(g, "self").sum() == 14


# --- Schichten gegen dichte Vergleichsrechnung ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("directed", [False, True])
def test_gat_and_gatv2_layers_agree_with_dense_reference_computations(directed):
    Am = random_graph(13, 0.25, 5, directed)
    g = A.edges_of(Am)
    mask = (Am + np.eye(13)) > 0
    rng = np.random.default_rng(6)
    H = rng.normal(size=(13, 5))
    heads, f = 3, 2
    W, a_s, a_d = rng.normal(size=(5, heads * f)), rng.normal(size=(heads, f)), rng.normal(size=(heads, f))
    assert A.gat_layer(H, W, a_s, a_d, g)[0] == pytest.approx(dense_gat(H, W, a_s, a_d, mask))
    Wl, Wr, a = rng.normal(size=(5, heads * f)), rng.normal(size=(5, heads * f)), rng.normal(size=(heads, f))
    assert A.gatv2_layer(H, Wl, Wr, a, g)[0] == pytest.approx(dense_gatv2(H, Wl, Wr, a, mask))


# --- Handrechnung: statisch gegen dynamisch ------------------------------------------------------------------------------------------------


def four_node_graph():
    """Knoten 0, 1 = Fragende mit (2, 0) und (0, 2); Knoten 2, 3 = Lager A (1, -1) und B (-1, 1); beide Lager senden an beide Fragenden."""
    X = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, -1.0], [-1.0, 1.0]])
    Am = np.zeros((4, 4))
    Am[0, 2] = Am[0, 3] = Am[1, 2] = Am[1, 3] = 1.0
    return X, A.edges_of(Am)


def test_gatv2_by_hand_the_ranking_flips_with_the_query():
    """Mit W_l = W_r = I, a = (1, 1): e(0, A) = LeakyReLU(3) + LeakyReLU(-1) = 3 - 0,2 = 2,8; e(0, B) = LeakyReLU(1) + LeakyReLU(1) = 2 -> A vor B.
    e(1, A) = LeakyReLU(1) + LeakyReLU(1) = 2; e(1, B) = LeakyReLU(-1) + LeakyReLU(3) = 2,8 -> B vor A."""
    X, g = four_node_graph()
    I2 = np.eye(2)
    out, cache = A.gatv2_layer(X, I2, I2, np.array([[1.0, 1.0]]), g)
    al = cache[5][:, 0]
    pos = {(int(d), int(s)): k for k, (d, s) in enumerate(zip(g.dst, g.src))}
    assert al[pos[(0, 2)]] > al[pos[(0, 3)]] and al[pos[(1, 3)]] > al[pos[(1, 2)]]
    # Zeile 0: Selbstschleife e(0,0) = LeakyReLU(4) + LeakyReLU(0) = 4; Softmax über (4, 2,8, 2)
    row0 = np.exp([4.0, 2.8, 2.0])
    assert [al[pos[(0, 0)]], al[pos[(0, 2)]], al[pos[(0, 3)]]] == pytest.approx(row0 / row0.sum())


def test_gat_ranking_can_never_flip_for_any_parameters():
    X, g = four_node_graph()
    pos = {(int(d), int(s)): k for k, (d, s) in enumerate(zip(g.dst, g.src))}
    flips = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        W, a_s, a_d = rng.normal(size=(2, 4)), rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
        al = A.gat_layer(X, W, a_s, a_d, g)[1][3]
        for h in range(2):
            flips += (al[pos[(0, 2)], h] > al[pos[(0, 3)], h]) != (al[pos[(1, 2)], h] > al[pos[(1, 3)], h])
    assert flips == 0


# --- Gradienten ---------------------------------------------------------------------------------------------------------------------------------


def numeric_grad_check(kind, heads, f, directed, gcn_mode="sym"):
    Am = random_graph(18, 0.2, 7, directed)
    rng = np.random.default_rng(8)
    X = rng.normal(size=(18, 4))
    y = rng.integers(0, 3, 18)
    tr = np.zeros(18, dtype=bool)
    tr[:9] = True
    g = A.edges_of(Am)
    w = A.gcn_weights(g, gcn_mode) if kind in ("gcn", "mlp") else None
    p = A.init_params(kind, 4, heads, f, 3, 2, hidden=6)
    _, gr = A.loss_and_grads(kind, p, X, y, tr, g, 5e-4, w)
    for k in p:
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p[k].shape)
            h = 1e-6
            pp = {a: b.copy() for a, b in p.items()}
            pm = {a: b.copy() for a, b in p.items()}
            pp[k][idx] += h
            pm[k][idx] -= h
            num = (A.loss_and_grads(kind, pp, X, y, tr, g, 5e-4, w)[0] - A.loss_and_grads(kind, pm, X, y, tr, g, 5e-4, w)[0]) / (2 * h)
            assert gr[k][idx] == pytest.approx(num, rel=1e-4, abs=1e-8)


@pytest.mark.parametrize("heads,f", [(1, 3), (2, 3), (3, 2)])
@pytest.mark.parametrize("directed", [False, True])
def test_gradients_of_gat_and_gatv2_agree_with_central_differences(heads, f, directed):
    numeric_grad_check("gat", heads, f, directed)
    numeric_grad_check("gatv2", heads, f, directed)


@pytest.mark.parametrize("mode,directed", [("sym", False), ("mean", True)])
def test_gradients_of_gcn_agree_with_central_differences_also_for_directed_graphs(mode, directed):
    numeric_grad_check("gcn", 1, 1, directed, gcn_mode=mode)


# --- Training ---------------------------------------------------------------------------------------------------------------------------------


def test_training_is_reproducible_reduces_the_loss_and_the_mlp_ignores_edges():
    g = S.generate_areas(120, 3, 5, 1.0, 0.0, 3)
    tr = S.split_areas(g.y, 6, 3)
    te = ~tr
    for kind in ("gcn", "gat", "gatv2"):
        a = A.train(kind, g.A, g.X, g.y, tr, te, seed=2, epochs=80, heads=2, f=4)
        b = A.train(kind, g.A, g.X, g.y, tr, te, seed=2, epochs=80, heads=2, f=4)
        assert all(np.array_equal(a.params[k], b.params[k]) for k in a.params) and a.history["loss"][-1] < 0.6 * a.history["loss"][0]
        assert float((A.predict(a, g.A, g.X)[te] == g.y[te]).mean()) == pytest.approx(a.history["test_acc"][-1])
    m1 = A.train("mlp", g.A, g.X, g.y, tr, te, seed=2, epochs=30)
    m2 = A.train("mlp", np.zeros_like(g.A), g.X, g.y, tr, te, seed=2, epochs=30)
    assert all(np.allclose(m1.params[k], m2.params[k]) for k in m1.params)


def test_recorded_predictions_match_the_final_prediction():
    g = S.generate_areas(60, 3, 5, 1.0, 0.0, 4)
    tr = S.split_areas(g.y, 4, 4)
    m = A.train("gatv2", g.A, g.X, g.y, tr, ~tr, seed=1, epochs=10, heads=2, f=3, record_pred=True)
    assert m.history["pred"].shape == (10, 60) and np.array_equal(m.history["pred"][-1], A.predict(m, g.A, g.X))
