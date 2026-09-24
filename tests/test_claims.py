"""Jede Zahl aus README und PRESET_HELP als Test. Einzelläufe nur mit Strukturgrenzen (plattformrobust), Mehr-Seed-Zahlen mit großzügigen Bändern."""

import pytest

import g2_evaluation as E
import g2_presets as P


def _preset(name):
    p = P.PRESETS[name]
    return E.analyse(E.Settings(p["K"], p["n_cust"], p["known"], p["heads"], p["dim"], p["seed"]))


def test_standard_preset_gatv2_solves_what_gat_and_gcn_do_not():
    a = _preset("Standardfall (6 Lager, 4 Köpfe)")
    assert a.acc["gatv2"] >= 0.85 and a.acc["gatv2"] - a.acc["gat"] > 0.05 and a.acc["gcn"] < a.chance + 0.2
    assert a.invariance["gat"] == 1.0 and a.invariance["gatv2"] < 0.6


def test_two_depots_are_solved_by_both_attention_models():
    a = _preset("Nur 2 Lager")
    assert a.acc["gat"] >= 0.9 and a.acc["gatv2"] >= 0.9 and a.acc["gcn"] < 0.9


def test_as_many_depots_as_heads_gat_nearly_keeps_up():
    a = _preset("So viele Lager wie Köpfe (4)")
    assert a.acc["gat"] >= 0.85 and a.acc["gatv2"] >= 0.9 and a.acc["gatv2"] - a.acc["gat"] < 0.2


def test_eight_depots_are_hard_but_gatv2_still_leads():
    a = _preset("Viele Lager (8)")
    assert a.acc["gatv2"] - a.acc["gat"] > 0.05 and a.acc["gatv2"] < 0.97


def test_one_head_gatv2_still_finds_the_match_while_the_ranking_always_changes():
    a = _preset("Nur ein Kopf")
    assert a.acc["gatv2"] - a.acc["gat"] > 0.1 and a.acc["gatv2"] >= 0.8 and a.invariance["gatv2"] < 0.1


def test_few_known_customers_make_it_data_hungry():
    a = _preset("Wenige bekannte Kunden (30 %)")
    full = _preset("Standardfall (6 Lager, 4 Köpfe)")
    assert a.acc["gatv2"] < full.acc["gatv2"] - 0.2 and a.acc["gatv2"] < 0.8


@pytest.fixture(scope="module")
def keys_rows():
    return {r["K"]: r for r in E.keys_experiment()}


def test_keys_experiment_static_attention_breaks_once_depots_exceed_heads(keys_rows):
    r = keys_rows
    assert r[2]["gat"] >= 0.9 and r[2]["gatv2"] >= 0.9
    assert r[6]["gatv2"] - r[6]["gat"] > 0.1 and r[8]["gatv2"] - r[8]["gat"] > 0.1
    assert r[8]["gat"] < 0.7 and r[8]["gatv2"] < r[2]["gatv2"] - 0.1
    assert all(0.4 <= r[k]["gcn"] <= 0.8 for k in r) and r[6]["gcn"] < r[6]["gat"] + 0.1


@pytest.fixture(scope="module")
def head_rows():
    return {r["heads"]: r for r in E.heads_experiment()}


def test_heads_experiment_more_heads_do_not_reliably_rescue_gat(head_rows):
    r = head_rows
    assert all(r[h]["gatv2"] >= 0.8 for h in r) and all(r[h]["gat"] <= 0.9 for h in r)
    assert r[8]["gat"] - r[1]["gat"] < 0.25 and all(r[h]["diff"] > 0.05 for h in r)


@pytest.fixture(scope="module")
def wrong_rows():
    return E.wrong_experiment()


def test_wrong_edges_gatv2_is_no_better_than_gat_and_not_better_than_gcn(wrong_rows):
    rows = wrong_rows
    assert all(abs(r["diff_v2_gat"]) < 0.08 for r in rows) and all(r["diff_v2_gcn"] < 0.03 for r in rows)
    assert rows[0]["homophily"] > 0.85 and rows[-1]["homophily"] < 0.55
    assert rows[-1]["mlp"] > rows[-1]["gatv2"] + 0.05 and rows[0]["gatv2"] > rows[0]["mlp"] + 0.1
