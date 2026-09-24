"""Vehikel E und D, Auswertung: Aufbau der Sterne, Zuordnung der Kanten, Aufmerksamkeit je Lager (Handrechnung), Interventionstest, Experimentstruktur."""

import numpy as np
import pytest

import g2_algorithm as A
import g2_constants as C
import g2_evaluation as E
import g2_scenario as S


def test_lookup_structure_and_label_rule():
    lk = S.generate_lookup(5, 30, seed=2)
    K = 5
    assert lk.X.shape == (30 * 6, 2 * K + C.N_CLASSES) and lk.A.sum() == 30 * K
    for c in range(30):
        base = lk.cust[c]
        assert sorted(lk.keys[c]) == list(range(K)) and lk.X[base, :K].sum() == 1 and lk.X[base, lk.query[c]] == 1
        slot = int(np.flatnonzero(lk.keys[c] == lk.query[c])[0])
        assert lk.y[base] == lk.y[base + 1 + slot] and lk.A[base, base + 1 + slot] == 1
        assert lk.A[base + 1 + slot, base] == 0                                      # gerichtet: das Lager erfährt die Anfrage nicht
        assert lk.X[base + 1: base + 1 + K, :K].sum() == 0 and lk.X[base, K:].sum() == 0


def test_lookup_is_reproducible_and_seeds_differ():
    a, b, c = S.generate_lookup(4, 20, 1), S.generate_lookup(4, 20, 1), S.generate_lookup(4, 20, 2)
    assert np.array_equal(a.X, b.X) and not np.array_equal(a.X, c.X)


def test_masks_only_mark_customers_and_partition_them():
    lk = S.generate_lookup(4, 40, 1)
    tr, te = S.lookup_masks(lk, 0.75, 1)
    assert tr.sum() == 30 and te.sum() == 10 and not (tr & te).any() and set(np.flatnonzero(tr | te)) == set(lk.cust)


def test_with_query_replaces_only_the_customer_features():
    lk = S.generate_lookup(4, 20, 1)
    X = S.with_query(lk, np.full(20, 2))
    assert np.all(X[lk.cust, 2] == 1) and X[lk.cust, :4].sum() == 20 and np.array_equal(np.delete(X, lk.cust, axis=0), np.delete(lk.X, lk.cust, axis=0))


def test_depot_edges_point_at_the_depots_of_the_right_customer():
    lk = S.generate_lookup(4, 12, 3)
    g = A.edges_of(lk.A)
    idx = E.depot_edges(g, lk) + 1
    for c in range(12):
        assert list(g.dst[idx[c]]) == [lk.cust[c]] * 4 and list(g.src[idx[c]]) == [lk.cust[c] + 1 + d for d in range(4)]
    alpha = np.random.default_rng(0).random((g.n_edges, 2))
    assert E.depot_attention(alpha, g, lk).shape == (12, 4, 2)


def test_attention_metrics_by_hand():
    """Wenn die Aufmerksamkeit jedes Kunden ganz auf dem passenden Lager liegt, ist die Trefferquote 1 und M die Einheitsmatrix (mit Trefferquote 0 bei der Gegenprobe)."""
    lk = S.generate_lookup(3, 24, 1)
    g = A.edges_of(lk.A)
    alpha = np.full((g.n_edges, 1), 1e-6)
    idx = E.depot_edges(g, lk) + 1
    for c in range(24):
        alpha[idx[c, int(np.flatnonzero(lk.keys[c] == lk.query[c])[0])], 0] = 1.0
    hit, M = E.attention_metrics(alpha, g, lk, np.ones(24, dtype=bool))
    assert hit == 1.0 and np.diag(M) == pytest.approx(np.ones(3), abs=1e-4) and (M - np.diag(np.diag(M))).max() < 1e-4
    wrong = np.full((g.n_edges, 1), 1e-6)
    for c in range(24):
        wrong[idx[c, (int(np.flatnonzero(lk.keys[c] == lk.query[c])[0]) + 1) % 3], 0] = 1.0
    assert E.attention_metrics(wrong, g, lk, np.ones(24, dtype=bool))[0] == 0.0


def test_rank_invariance_is_exactly_one_for_gat_and_lower_for_gatv2():
    a = E.analyse(E.Settings(K=4, n_cust=60, known=0.75, heads=2, dim=8, seed=1))
    assert a.invariance["gat"] == 1.0 and a.invariance["gatv2"] < 0.9
    assert set(a.acc) == {"gcn", "gat", "gatv2"} and all(0 <= v <= 1 for v in a.acc.values())
    assert np.allclose(a.M["gatv2"].sum(axis=1), 1.0, atol=1e-6)


def test_analysis_is_cached_and_deterministic():
    s = E.Settings(K=3, n_cust=40, heads=2, dim=8, seed=4)
    assert E.analyse(s) is E.analyse(s)


def test_areas_homophily_and_rewiring_keep_the_edge_count():
    g0 = S.generate_areas(wrong=0.0, seed=1)
    g6 = S.generate_areas(wrong=0.6, seed=1)
    assert g0.A.sum() == g6.A.sum() and g6.edge_homophily() < g0.edge_homophily() - 0.2
    assert np.array_equal(g0.A, g0.A.T) and np.trace(g6.A) == 0 and np.array_equal(g6.A, g6.A.T)


def test_experiments_return_consistent_rows(monkeypatch):
    rows = E.keys_experiment(levels=(2, 3), seeds=(0,), n_cust=40)
    assert [r["K"] for r in rows] == [2, 3] and all(r["diff"] == pytest.approx(r["gatv2"] - r["gat"]) for r in rows)
    rows = E.heads_experiment(levels=(1, 2), seeds=(0,), K=3, n_cust=40)
    assert [r["heads"] for r in rows] == [1, 2] and all(r["diff"] == pytest.approx(r["gatv2"] - r["gat"]) for r in rows)
    monkeypatch.setattr(C, "AREA_EPOCHS", 20)
    rows = E.wrong_experiment(levels=(0.0, 0.6), seeds=(0, 1))
    assert rows[0]["homophily"] > rows[1]["homophily"] and all(0 <= r["v2_wins_vs_gcn"] <= 2 for r in rows)
    assert all(r["diff_v2_gcn"] == pytest.approx(r["gatv2"] - r["gcn"]) for r in rows)
