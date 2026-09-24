"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import g2_constants as C
import g2_evaluation as E
import g2_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        assert (p["n_cust"] - C.CUST_MIN) % C.CUST_STEP == 0
        k = (p["known"] - C.KNOWN_MIN) / C.KNOWN_STEP
        assert abs(k - round(k)) < 1e-9
        assert p["dim"] in C.DIM_OPTIONS


def test_default_preset_equals_the_default_settings():
    p = P.PRESETS["Standardfall (6 Lager, 4 Köpfe)"]
    assert E.Settings(p["K"], p["n_cust"], p["known"], p["heads"], p["dim"], p["seed"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("k_slider") == (C.K_MIN, C.K_MAX) and set(P.STEPS) == {"cust_slider", "known_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
