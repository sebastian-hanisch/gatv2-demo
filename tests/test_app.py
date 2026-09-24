"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Kundenwahl, Würfel-Knopf, Permalink-Grenzen, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import g2_constants as C
import g2_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_shows_gatv2_ahead_and_all_charts():
    at = _run()
    _ok(at)
    assert at.metric and any("Dynamische Aufmerksamkeit gewinnt" in s.value for s in at.success)
    assert len(at.get("plotly_chart")) == 3


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]
    assert at.get("plotly_chart")


def test_customer_selection_survives_shrinking_n():
    at = _run(cust_slider=400, cust_select=399)
    _ok(at)
    at.slider(key="cust_slider").set_value(100).run()
    _ok(at)
    assert at.session_state["cust_select"] < 100 and any("sucht Artikelcode" in c.value for c in at.caption)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Lager-Suche generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["cust"] = "9999"
    at.query_params["known"] = "0.33"
    at.query_params["k"] = "abc"
    at.query_params["heads"] = "99"
    at.query_params["dim"] = "12"
    at.run()
    _ok(at)
    assert at.session_state["cust_slider"] == C.CUST_MAX and at.session_state["known_slider"] == 0.35 and at.session_state["k_slider"] == C.DEFAULT_K
    assert at.session_state["heads_slider"] == C.HEADS_MAX and at.session_state["dim_select"] == C.DEFAULT_DIM


@pytest.mark.parametrize("kw", [dict(k_slider=C.K_MIN), dict(k_slider=C.K_MAX, cust_slider=C.CUST_MIN), dict(known_slider=C.KNOWN_MIN), dict(known_slider=C.KNOWN_MAX), dict(heads_slider=C.HEADS_MAX, dim_select=8),
                                dict(heads_slider=C.HEADS_MIN, k_slider=3)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_keys_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "K_LEVELS", (2, 3))
    monkeypatch.setattr(C, "EXP_SEEDS_LOOKUP", (0,))
    monkeypatch.setattr(C, "EXP_CUST", 40)
    at = _run()
    next(b for b in at.button if b.key == "keys_start").click().run()
    _ok(at)
    assert at.session_state["keys_on"] and any("2 Lager: GAT" in w.value for w in at.warning)


def test_heads_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "HEAD_LEVELS", (1, 2))
    monkeypatch.setattr(C, "EXP_SEEDS_LOOKUP", (0,))
    monkeypatch.setattr(C, "EXP_CUST", 40)
    at = _run()
    next(b for b in at.button if b.key == "heads_start").click().run()
    _ok(at)
    assert at.session_state["heads_on"] and any("1 Köpfe: GAT" in w.value for w in at.warning)


def test_wrong_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "WRONG_LEVELS", (0.0, 0.6))
    monkeypatch.setattr(C, "EXP_SEEDS_AREA", (0, 1))
    monkeypatch.setattr(C, "AREA_EPOCHS", 20)
    at = _run()
    next(b for b in at.button if b.key == "wrong_start").click().run()
    _ok(at)
    assert at.session_state["wrong_on"] and any("GATv2 minus GAT in Punkten" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{pts(" not in el.value
