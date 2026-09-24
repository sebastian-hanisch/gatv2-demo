"""GATv2 - dynamische Aufmerksamkeit - wonach der Kunde fragt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Graph-Neural-Network-Linie der "Konzepte"-Reihe (Nachfolger von gat-demo): GAT bewertet einen Nachbarn als Summe aus einem Anteil des Fragenden und einem des Nachbarn - die Rangfolge der Nachbarn ist für jeden
Fragenden dieselbe. GATv2 verschiebt die Nichtlinearität vor das Skalarprodukt; dann darf die Rangfolge vom Fragenden abhängen.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import g2_constants as C
from g2_evaluation import Settings, analyse, heads_experiment, keys_experiment, wrong_experiment
from g2_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from g2_visualization import build_attention_matrices, build_curves, build_heads, build_keys, build_star, build_wrong

st.set_page_config(page_title="GATv2 – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


def pts(x, se=None, digits=1):
    s = f"{'+' if x >= 0 else '−'}{de(abs(100 * x), digits)}"
    return s + (f" ± {de(100 * se, digits)}" if se is not None else "")


@st.cache_data(show_spinner=False)
def _keys(levels, seeds, n_cust, heads):
    return keys_experiment(levels=levels, seeds=seeds, n_cust=n_cust, heads=heads)


@st.cache_data(show_spinner=False)
def _heads(levels, seeds, K, n_cust):
    return heads_experiment(levels=levels, seeds=seeds, K=K, n_cust=n_cust)


@st.cache_data(show_spinner=False)
def _wrong(levels, seeds):
    return wrong_experiment(levels=levels, seeds=seeds)


st.title("🔍 GATv2 – Aufmerksamkeit, die vom Fragenden abhängt")
st.markdown(
    """
Im Vorgänger-Stück lernte ein **GAT**, welche Nachbarn wichtig sind - aber jeder Kunde ordnet seine Nachbarn **in derselben Reihenfolge**, weil die Bewertung eine Summe aus einem Anteil des Fragenden und einem des Nachbarn ist
("statische Aufmerksamkeit"). **GATv2** (Brody/Alon/Yahav 2022) rechnet stattdessen $a \\cdot \\mathrm{LeakyReLU}(W_l h_i + W_r h_j)$: die Nichtlinearität steht **vor** dem Skalarprodukt, und die Rangfolge darf vom Fragenden abhängen.
Die Demo zeigt das an einer **Lager-Suche**, die genau das verlangt: jeder Kunde sucht unter seinen Lagern das mit seinem Artikelcode - und misst ehrlich, was es im Liefergebiet bei falschen Kanten bringt (dort: nicht viel).
"""
)
st.caption(
    "Drittes Stück der **Graph-Neural-Network-Linie** der \"Konzepte\"-Reihe, Nachfolger von **gat-demo**; das Netz ist von Grund auf in numpy geschrieben (Kantenlisten, Rückwärtsrechnung von Hand). "
    "**Bezug zu OR:** Zuordnungen wie \"welches Lager bedient diesen Artikel\" hängen vom Artikel ab - ein Modell, das Lager unabhängig von der Anfrage bewertet, kann das nicht abbilden."
)

with st.expander("So funktioniert die Lager-Suche und der Unterschied GAT / GATv2", expanded=True):
    st.markdown(
        """
1. **Aufgabe.** Ein Kunde kennt seinen **Artikelcode** $q$; seine $K$ Lager führen je genau einen Code (jedes Mal neu gemischt) und liegen in einer **Servicezone**. Gesucht: die Zone des Lagers, dessen Code zu $q$ passt.
   Die Lager senden an den Kunden, aber der Kunde nicht an die Lager - ein Lager weiß nicht, wonach gefragt wird.
2. **GCN** mittelt alle Lager: der Mittelwert der Zonen ist derselbe für jede Anfrage - er kann nicht wissen, welches zählt.
3. **GAT** bewertet Lager $j$ für Kunde $i$ mit $\\mathrm{LeakyReLU}(a_E \\cdot Wh_i + a_S \\cdot Wh_j)$. Der Anteil $a_S \\cdot Wh_j$ hängt nur vom Lager ab, deshalb ist die **Rangfolge** der Lager für jeden Kunden gleich: ein Kopf kann
   "Lager mit Code 3" immer bevorzugen, aber nicht "das Lager, das zu meiner Anfrage passt". Mit ebenso vielen Köpfen wie Lagern lässt sich das im Prinzip umgehen (je ein fester "Zeiger" auf einen Code); ob das Training solche Zeiger findet, misst das zweite Experiment.
4. **GATv2** bewertet mit $a \\cdot \\mathrm{LeakyReLU}(W_l h_i + W_r h_j)$ - Anfrage und Lager gehen gemeinsam durch die Nichtlinearität, so kann ein einzelner Kopf "passt" erkennen.
5. **Interventionstest.** Man ändert bei den Kunden die Anfrage und prüft, ob sich die Rangfolge der Lager (nach Aufmerksamkeit) ändert: bei GAT nie, bei GATv2 oft.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    K = st.slider("Lager je Kunde", *bounds("k_slider"), key="k_slider", help="Zahl der Lager (= Zahl der möglichen Artikelcodes), unter denen der Kunde das passende suchen muss.")
    n_cust = st.slider("Kunden", *bounds("cust_slider"), key="cust_slider", step=C.CUST_STEP, help="Zahl der Kunden (je mit eigenen Lagern). Mehr Kunden = mehr Trainingsbeispiele, aber längere Rechenzeit.")
    known = st.slider("Anteil bekannter Kunden", *bounds("known_slider"), key="known_slider", step=C.KNOWN_STEP, help="Anteil der Kunden, deren Zone dem Netz beim Training verraten wird; die übrigen dienen der Prüfung.")
    heads = st.slider("Köpfe", *bounds("heads_slider"), key="heads_slider", help="Zahl der Aufmerksamkeitsköpfe der ersten Schicht (GAT und GATv2).")
    dim = st.selectbox("Merkmale je Kopf", C.DIM_OPTIONS, key="dim_select", help="Breite jedes Kopfes der ersten Schicht.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Anfragen, Lager, Zonen, bekannte Kunden und Anfangsgewichte fest.")
    st.button("🎲 Neue Lager-Suche generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"k_slider": int(K), "cust_slider": int(n_cust), "known_slider": round(float(known), 2), "heads_slider": int(heads), "dim_select": int(dim), "seed_input": int(seed)})

settings = Settings(int(K), int(n_cust), round(float(known), 2), int(heads), int(dim), int(seed))
with st.spinner("Trainiere GCN, GAT und GATv2 (einige Sekunden)..."):
    a = analyse(settings)
lk = a.lookup

# --- Ergebnis ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wer findet das passende Lager?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("GCN (Mittel über alle Lager)", pct(a.acc["gcn"], 1), help="Alle Lager zählen nach den Graden; ohne Wissen, welches zur Anfrage passt.")
m2.metric("GAT (statisch)", pct(a.acc["gat"], 1), delta=f"{pts(a.acc['gat'] - a.acc['gcn'])} Punkte gegen GCN", delta_color="off", help="Lernbare Aufmerksamkeit, aber mit statischer Rangfolge der Lager.")
m3.metric("GATv2 (dynamisch)", pct(a.acc["gatv2"], 1), delta=f"{pts(a.acc['gatv2'] - a.acc['gat'])} Punkte gegen GAT", delta_color="off", help="Nichtlinearität vor dem Skalarprodukt: die Rangfolge darf von der Anfrage abhängen.")
m4.metric("Raten", pct(a.chance, 1), help="Anteil der häufigsten Zone unter den unbekannten Kunden.")
st.plotly_chart(build_curves(a), width="stretch", key="curves_chart")
n_test = int(a.test_mask.sum())
if a.acc["gatv2"] - a.acc["gat"] > 0.10:
    st.success(f"✅ Dynamische Aufmerksamkeit gewinnt: GATv2 {pct(a.acc['gatv2'], 1)} gegen GAT {pct(a.acc['gat'], 1)} und GCN {pct(a.acc['gcn'], 1)} auf {n_test} unbekannten Kunden. Bei {lk.K} Lagern und {settings.heads} Köpfen braucht "
               "die Suche eine Bewertung, die von der Anfrage abhängt.")
elif a.acc["gat"] > 0.9 and a.acc["gatv2"] > 0.9:
    st.info(f"Beide lösen die Suche: GAT {pct(a.acc['gat'], 1)}, GATv2 {pct(a.acc['gatv2'], 1)}. Bei {lk.K} Lagern reichen {settings.heads} Köpfe, um jeden Code fest zu bevorzugen - die statische Aufmerksamkeit kommt mit Zeigern aus.")
elif a.acc["gatv2"] < a.chance + 0.15:
    st.warning(f"⚠️ Keines der Netze löst die Suche: GATv2 {pct(a.acc['gatv2'], 1)} liegt nahe am Raten ({pct(a.chance, 1)}). Mit {int(a.train_mask.sum())} bekannten Kunden ist die Zuordnung zu schwer zu lernen.")
else:
    st.info(f"GAT {pct(a.acc['gat'], 1)}, GATv2 {pct(a.acc['gatv2'], 1)}, GCN {pct(a.acc['gcn'], 1)} (Raten {pct(a.chance, 1)}).")

st.markdown("---")

# --- Aufmerksamkeit ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wohin schaut die Aufmerksamkeit?")
st.plotly_chart(build_attention_matrices(a), width="stretch", key="matrix_chart")
st.caption(
    f"Jede Zeile: Kunden mit demselben Anfrage-Schlüssel; jede Spalte: der Schlüssel des Lagers. Auf der Diagonalen liegt das passende Lager. Die Aufmerksamkeit (Mittel der {settings.heads} Köpfe der ersten Schicht) liegt bei GATv2 bei "
    f"{pct(a.hit['gatv2'])} der unbekannten Kunden am höchsten auf dem passenden Lager, bei GAT bei {pct(a.hit['gat'])} (Zufall: {pct(1 / lk.K)}). "
)
c1, c2 = st.columns(2)
c1.metric("GAT: Rangfolge bleibt bei anderer Anfrage gleich", pct(a.invariance["gat"]), help="Interventionstest: Anteil der (Kunde, Kopf, andere Anfrage)-Fälle mit derselben Rangfolge der Lager wie bei der echten Anfrage.")
c2.metric("GATv2: Rangfolge bleibt bei anderer Anfrage gleich", pct(a.invariance["gatv2"]), help="Dasselbe für GATv2.")
st.caption("Beim GAT ist die Rangfolge der Lager **immer** dieselbe, auch wenn man die Anfrage austauscht - die Bewertung ist eine Summe. Beim GATv2 ändert sie sich mit der Anfrage; das ist die Voraussetzung dafür, das passende Lager zu finden.")

st.markdown("##### Ein Kunde im Detail")
if "cust_select" not in st.session_state or st.session_state["cust_select"] >= lk.n_cust:
    st.session_state["cust_select"] = 0
cnum = int(st.number_input("Kunde Nr.", 0, lk.n_cust - 1, key="cust_select", step=1, help="Nummer eines Kunden (0 bis Zahl der Kunden - 1)."))
st.plotly_chart(build_star(a, cnum), width="stretch", key="star_chart")
st.caption(
    f"Kunde {cnum} sucht Artikelcode {lk.query[cnum]}. Die Kreise sind seine Lager (Zahl = Code, Farbe = Zone); der grüne Ring markiert das passende, Strichbreiten sind die Aufmerksamkeit (Kopf-Mittel, erste Schicht). "
    f"Wahre Zone: {C.CLASS_NAMES[lk.y[lk.cust[cnum]]]}; {'bekannter Trainingskunde' if a.train_mask[lk.cust[cnum]] else 'unbekannter Kunde'}."
)

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viele Lager verträgt die statische Aufmerksamkeit?")
st.caption(f"{C.EXP_CUST} Kunden, 75 % bekannt, 4 Köpfe, Zahl der Lager {', '.join(str(k) for k in C.K_LEVELS)}; Mittel über {len(C.EXP_SEEDS_LOOKUP)} feste Seeds (Fehlerbalken: Standardfehler). Dauer etwa zwei Minuten.")
if st.button("Zahl der Lager durchrechnen", key="keys_start"):
    st.session_state["keys_on"] = True
if st.session_state.get("keys_on"):
    with st.spinner("Rechne (etwa zwei Minuten)..."):
        rows_k = _keys(C.K_LEVELS, C.EXP_SEEDS_LOOKUP, C.EXP_CUST, C.DEFAULT_HEADS)
    st.plotly_chart(build_keys(rows_k), width="stretch", key="keys_chart")
    parts = "; ".join(f"{r['K']} Lager: GAT {pct(r['gat'])}, GATv2 {pct(r['gatv2'])} ({pts(r['diff'], r['diff_se'])} Punkte)" for r in rows_k)
    rmax = max(rows_k, key=lambda r: r["diff"])
    st.warning(
        f"**Befund:** {parts}; GCN {pct(min(r['gcn'] for r in rows_k))} bis {pct(max(r['gcn'] for r in rows_k))} (Raten etwa {pct(np.mean([r['chance'] for r in rows_k]))}). Der Vorsprung von GATv2 ist bei {rmax['K']} Lagern am größten ({pts(rmax['diff'], rmax['diff_se'])} Punkte); "
        "bei so vielen Lagern wie Köpfen (oder weniger) kommt auch GAT zurecht, weil jeder Kopf einen Code fest bevorzugen kann."
    )

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Helfen mehr Köpfe dem GAT?")
st.caption(f"{C.DEFAULT_K} Lager, {C.EXP_CUST} Kunden, 75 % bekannt; Zahl der Köpfe {', '.join(str(h) for h in C.HEAD_LEVELS)}; {len(C.EXP_SEEDS_LOOKUP)} feste Seeds. Dauer etwa zwei bis drei Minuten.")
if st.button("Zahl der Köpfe durchrechnen", key="heads_start"):
    st.session_state["heads_on"] = True
if st.session_state.get("heads_on"):
    with st.spinner("Rechne (etwa zwei bis drei Minuten)..."):
        rows_h = _heads(C.HEAD_LEVELS, C.EXP_SEEDS_LOOKUP, C.DEFAULT_K, C.EXP_CUST)
    st.plotly_chart(build_heads(rows_h), width="stretch", key="heads_chart")
    parts = "; ".join(f"{r['heads']} Köpfe: GAT {pct(r['gat'])}, GATv2 {pct(r['gatv2'])}" for r in rows_h)
    r1, rl = rows_h[0], rows_h[-1]
    st.warning(
        f"**Befund:** {parts}. GAT verbessert sich von {r1['heads']} auf {rl['heads']} Köpfe um {pts(rl['gat'] - r1['gat'])} Punkte; GATv2 kommt schon mit einem Kopf auf {pct(r1['gatv2'])}. "
        "Theoretisch genügen ebenso viele Köpfe wie Lager, damit GAT die Suche lösen kann; ob das Training solche festen Zeiger tatsächlich findet, zeigt die Kurve."
    )

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Und bei falschen Kanten im Liefergebiet?")
st.caption(f"Vehikel des Vorgängers (200 Kunden, 3 Gebietstypen, 20 Etiketten je Typ, keine unzuverlässigen Kunden); ein Anteil der Kanten wird durch zufällige ersetzt ({', '.join(pct(x) for x in C.WRONG_LEVELS)}); "
           f"{len(C.EXP_SEEDS_AREA)} feste Seeds. Dauer etwa eine halbe Minute.")
if st.button("Falsche Kanten durchrechnen", key="wrong_start"):
    st.session_state["wrong_on"] = True
if st.session_state.get("wrong_on"):
    with st.spinner("Rechne..."):
        rows_w = _wrong(C.WRONG_LEVELS, C.EXP_SEEDS_AREA)
    st.plotly_chart(build_wrong(rows_w), width="stretch", key="wrong_chart")
    parts = "; ".join(f"Homophilie {pct(r['homophily'])}: {pts(r['diff_v2_gat'], r['diff_v2_gat_se'])}" for r in rows_w)
    rm = rows_w[len(rows_w) // 2]
    st.warning(
        f"**Befund:** GATv2 minus GAT in Punkten - {parts}: kein nennenswerter Unterschied. Auch die dynamische Aufmerksamkeit schützt nicht vor falschen Kanten: bei Homophilie {pct(rm['homophily'])} liegt GATv2 mit {pct(rm['gatv2'], 1)} "
        f"hinter dem GCN ({pct(rm['gcn'], 1)}; {pts(rm['diff_v2_gcn'], rm['diff_v2_gcn_se'])} Punkte), bei der stärksten Störung ({pct(rows_w[-1]['homophily'])}) hinter dem MLP ohne Nachbarn ({pct(rows_w[-1]['mlp'], 1)}). "
        "Der theoretische Vorteil (Rangfolge abhängig vom Fragenden) zeigt sich in dieser Aufgabe nicht; woran das liegt, wurde nicht untersucht."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Beispiele, um "passt" zu lernen** | Dynamische Aufmerksamkeit ist ausdrucksstärker, aber datenhungrig: mit wenigen bekannten Kunden oder vielen Lagern lernen auch GATv2 und GAT die Suche nicht zuverlässig (Preset, Experimente). | – |
| **Die Aufgabe verlangt Abhängigkeit vom Fragenden** | Bei vielen Graphaufgaben (Homophilie, Gemeinschaften) reicht statische Aufmerksamkeit oder sogar ein Mittel; im Liefergebiet mit falschen Kanten bringt GATv2 nichts. | – |
| **Der Fragende sieht den Nachbarn nicht anders als andere Fragende** | Wenn sich die Bedeutung eines Nachbarn nur durch die Kombination erschließt, hilft nur Wechselwirkung (GATv2, Transformer) - oder viele Köpfe als feste Zeiger. | GATv2 (hier) |
| **Der ganze Graph ist beim Training bekannt** | Wie GCN und GAT transduktiv trainiert; neue Kunden oder sehr große Graphen brauchen Stichproben der Nachbarschaft. Hier nicht gemessen. | GraphSAGE |
| **Erzeugte Daten, wenige Seeds** | Die Lager-Suche ist ein synthetischer Prüfstein; die Experimente mitteln über 3 (Lager) bzw. 8 (Gebiet) Seeds, die Standardfehler sind groß. | – |
"""
)
st.caption("Die Linie: GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer (GraphSAGE, GIN und Graph Transformer noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kanten.** Eine Kante $j \to i$ heißt: $j$ sendet an $i$. Jeder Knoten hat eine Schleife. $\mathcal N(i)$ = Sender an $i$ (inklusive $i$). Die Softmax läuft über $\mathcal N(i)$.

**GAT.** $e_{ij} = \mathrm{LeakyReLU}_{0{,}2}\big(a_E \cdot Wh_i + a_S \cdot Wh_j\big)$. Weil $\mathrm{LeakyReLU}$ monoton ist und $a_E \cdot Wh_i$ für alle $j \in \mathcal N(i)$ derselbe Summand ist, ordnet $e_{ij}$ die Nachbarn
nach $a_S \cdot Wh_j$ - für jedes $i$ gleich (**statisch**). Ein Kopf kann also nur "Sender mit Eigenschaft X bevorzugen".

**GATv2.** $e_{ij} = a \cdot \mathrm{LeakyReLU}\big(W_l h_i + W_r h_j\big)$, Nachricht $W_r h_j$. Die Nichtlinearität wirkt auf die Summe: die Rangfolge kann vom Fragenden $i$ abhängen (**dynamisch**).

**Beide:** $\alpha_{ij} = \mathrm{softmax}_j(e_{ij})$, $h_i' = \sum_{j \in \mathcal N(i)} \alpha_{ij}\,(\text{Nachricht}_j)$, je Kopf; zwei Schichten (Köpfe × Merkmale mit ReLU, dann ein Kopf mit den Klassen-Logits); Kreuzentropie auf den bekannten Knoten,
Gewichtszerfall $5\cdot10^{-4}$ auf den Matrizen, Adam (0,01), 400 Epochen (Lager-Suche) bzw. 200 (Liefergebiet).

**Rückwärtsrechnung GATv2** (je Kopf, $\delta_i$ = Gradient an der Ausgabe von $i$): $\partial\alpha_{ij} = \delta_i \cdot W_r h_j$; Softmax: $\partial e_{ij} = \alpha_{ij}(\partial\alpha_{ij} - \sum_l \alpha_{il}\partial\alpha_{il})$; $\partial a = \sum_{ij} \partial e_{ij}\,z_{ij}$
mit $z_{ij} = \mathrm{LeakyReLU}(u_{ij})$, $u_{ij} = W_l h_i + W_r h_j$; $\partial u_{ij} = \partial e_{ij}\,a \odot \mathrm{LeakyReLU}'(u_{ij})$; $\partial (W_l h)_i = \sum_j \partial u_{ij}$, $\partial (W_r h)_j = \sum_i \partial u_{ij} + \sum_i \alpha_{ij}\delta_i$.

**GCN-Vergleich.** Mittel über die Eingangskanten (Lager-Suche, gerichtet) bzw. $D^{-1/2}(A+I)D^{-1/2}$ (Liefergebiet, ungerichtet).

Implementiert in `g2_algorithm.py` (Kantenlisten, Schichten, Adam), `g2_scenario.py` (Vehikel), `g2_evaluation.py` (Analyse, Interventionstest, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
