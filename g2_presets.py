"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. gat_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import g2_constants as C


def _dim(value):
    v = int(float(value))
    if v not in C.DIM_OPTIONS:
        raise ValueError(value)
    return v


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "cust_slider": SettingSpec("cust", int, C.DEFAULT_CUST, C.CUST_MIN, C.CUST_MAX),
    "known_slider": SettingSpec("known", float, C.DEFAULT_KNOWN, C.KNOWN_MIN, C.KNOWN_MAX),
    "heads_slider": SettingSpec("heads", int, C.DEFAULT_HEADS, C.HEADS_MIN, C.HEADS_MAX),
    "dim_select": SettingSpec("dim", _dim, C.DEFAULT_DIM),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"K": "k_slider", "n_cust": "cust_slider", "known": "known_slider", "heads": "heads_slider", "dim": "dim_select", "seed": "seed_input"}
STEPS = {"cust_slider": C.CUST_STEP, "known_slider": C.KNOWN_STEP}


def _p(**kw):
    base = {"K": C.DEFAULT_K, "n_cust": C.DEFAULT_CUST, "known": C.DEFAULT_KNOWN, "heads": C.DEFAULT_HEADS, "dim": C.DEFAULT_DIM, "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall (6 Lager, 4 Köpfe)": _p(),
    "Nur 2 Lager": _p(K=2),
    "So viele Lager wie Köpfe (4)": _p(K=4),
    "Viele Lager (8)": _p(K=8),
    "Nur ein Kopf": _p(heads=1),
    "Wenige bekannte Kunden (30 %)": _p(known=0.3),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall (6 Lager, 4 Köpfe)": "Seed 3: GCN 40 %, GAT 82 %, GATv2 96 % (Raten 38 %). Im Mittel über drei Seeds liegt GAT bei etwa 69 %, GATv2 bei 96 %. Die Rangfolge der Lager bleibt beim GAT bei anderer Anfrage immer gleich (100 %), beim GATv2 nur in etwa 29 % der Fälle.",
    "Nur 2 Lager": "Seed 3: GAT und GATv2 finden beide das passende Lager (100 %), GCN kommt nur auf 70 %. Mit zwei Lagern genügt ein fester Zeiger auf einen Code - die statische Aufmerksamkeit reicht.",
    "So viele Lager wie Köpfe (4)": "Seed 3: GAT 96 %, GATv2 100 %, GCN 64 %. Jeder der vier Köpfe kann einen Code fest bevorzugen; der Vorsprung des GATv2 ist klein.",
    "Viele Lager (8)": "Seed 3: GCN 46 %, GAT 54 %, GATv2 80 % (Raten 42 %). Mit acht Lagern und vier Köpfen wird die Suche auch für GATv2 schwer; im Mittel über drei Seeds GAT 45 %, GATv2 73 %.",
    "Nur ein Kopf": "Seed 3: GAT 66 %, GATv2 90 %, GCN 40 %. Ein einziger GATv2-Kopf kann bereits \"passt\" erkennen; die Rangfolge ändert sich mit der Anfrage in allen Fällen (0 % gleich).",
    "Wenige bekannte Kunden (30 %)": "Seed 3: GCN 39 %, GAT 49 %, GATv2 56 % (Raten 37 %). Nur 60 Kunden mit bekannter Zone reichen kaum, um \"passt\" zu lernen - die dynamische Aufmerksamkeit ist datenhungrig.",
}
