"""Plotly-Abbildungen der GATv2-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import g2_constants as C
import g2_evaluation as E

ZONE_COLORS = ["#4c78a8", "#e45756", "#54a24b"]
REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
WARN = "#f58518"
PURPLE = "#b279a2"
LINE_COLOR = "#4c78a8"
TEAL = "#17becf"
BAD = "#e45756"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_curves(a):
    E_ = len(a.models["gat"].history["test_acc"])
    xs = list(range(1, E_ + 1))
    fig = go.Figure()
    for kind, name, color in (("gcn", "GCN", LINE_COLOR), ("gat", "GAT", PURPLE), ("gatv2", "GATv2", TEAL)):
        fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.models[kind].history["test_acc"]], mode="lines", name=name, line=dict(color=color, width=2.5)))
    fig.add_hline(y=100 * a.chance, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Raten (häufigste Zone)", annotation_position="bottom right")
    fig.update_xaxes(title_text="Epoche")
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[0, 102])
    return _base(fig, 320)


def build_attention_matrices(a):
    """Mittlere Aufmerksamkeit (Kopf-Mittel) eines Kunden mit Anfrage-Schlüssel q (Zeile) auf das Lager mit Schlüssel k (Spalte). Diagonale = das passende Lager."""
    K = a.lookup.K
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"GAT: Trefferquote {100 * a.hit['gat']:.0f} %", f"GATv2: Trefferquote {100 * a.hit['gatv2']:.0f} %"), horizontal_spacing=0.12)
    zmax = max(a.M["gat"].max(), a.M["gatv2"].max())
    for col, kind in ((1, "gat"), (2, "gatv2")):
        fig.add_trace(go.Heatmap(z=a.M[kind], x=list(range(K)), y=list(range(K)), zmin=0, zmax=zmax, colorscale="Purples", showscale=col == 2,
                                 hovertemplate="Anfrage %{y}, Lager-Schlüssel %{x}: %{z:.2f}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="Schlüssel des Lagers", dtick=1, row=1, col=col)
        fig.update_yaxes(title_text="Anfrage-Schlüssel des Kunden" if col == 1 else None, dtick=1, autorange="reversed", row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_star(a, c):
    """Der Stern eines Kunden: Lager auf einem Kreis (Farbe = Zone, Beschriftung = Schlüssel), Strichbreite = Aufmerksamkeit (Kopf-Mittel, erste Schicht); grüner Ring = das passende Lager."""
    lk = a.lookup
    K = lk.K
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    px, py = np.cos(ang), np.sin(ang)
    base = lk.cust[c]
    keys = lk.keys[c]
    zones = np.array([int(np.argmax(lk.X[base + 1 + d, 2 * K:])) for d in range(K)])
    match = int(np.flatnonzero(keys == lk.query[c])[0])
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"GAT (Anfrage: Artikelcode {lk.query[c]})", f"GATv2 (Anfrage: Artikelcode {lk.query[c]})"), horizontal_spacing=0.06)
    for col, kind in ((1, "gat"), (2, "gatv2")):
        att = E.depot_attention(a.alpha[kind], a.graph, lk).mean(axis=2)[c]
        att = att / att.sum()
        scale = 16.0 / max(att.max(), 1e-9)
        for d in range(K):
            fig.add_trace(go.Scatter(x=[0, px[d]], y=[0, py[d]], mode="lines", line=dict(width=max(0.8, att[d] * scale), color="rgba(120,120,120,0.6)"), showlegend=False, hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=[px[match]], y=[py[match]], mode="markers", marker=dict(size=30, symbol="circle-open", color=GOOD, line=dict(width=3)), name="passendes Lager", showlegend=col == 1, hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers+text", marker=dict(size=18, color=[ZONE_COLORS[z] for z in zones], line=dict(color="white", width=1)), text=[f"{k}" for k in keys], textposition="middle center",
                                 textfont=dict(color="white", size=11), showlegend=False, hovertemplate="Schlüssel %{text}, Aufmerksamkeit %{customdata:.2f}<extra></extra>", customdata=att), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=16, symbol="star", color="black"), name="Kunde", showlegend=col == 1), row=1, col=col)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4])
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.05), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _lines(rows, xkey, xtitle, models, height=340, reverse=False, chance=False):
    spec = {"gcn": ("GCN", LINE_COLOR), "gat": ("GAT", PURPLE), "gatv2": ("GATv2", TEAL), "mlp": ("MLP (ohne Nachbarn)", REF_COLOR)}
    fig = go.Figure()
    xs = [100 * r[xkey] if xkey == "homophily" else r[xkey] for r in rows]
    for m in models:
        name, color = spec[m]
        fig.add_trace(go.Scatter(x=xs, y=[100 * r[m] for r in rows], error_y=dict(type="data", array=[100 * r[m + "_se"] for r in rows]), mode="lines+markers", name=name, line=dict(color=color, width=2.5)))
    if chance:
        fig.add_trace(go.Scatter(x=xs, y=[100 * r["chance"] for r in rows], mode="lines", name="Raten", line=dict(color=REF_COLOR, dash="dot", width=1.5)))
    fig.update_xaxes(title_text=xtitle, **({"autorange": "reversed"} if reverse else {}), dtick=1 if xkey in ("K", "heads") else None)
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[0, 102] if xkey != "homophily" else [30, 100])
    return _base(fig, height).update_layout(legend=dict(orientation="h", y=-0.3))


def build_keys(rows):
    return _lines(rows, "K", "Zahl der Lager je Kunde", ("gcn", "gat", "gatv2"), chance=True)


def build_heads(rows):
    return _lines(rows, "heads", "Zahl der Köpfe", ("gat", "gatv2"))


def build_wrong(rows):
    return _lines(rows, "homophily", "Anteil der Kanten zwischen Kunden desselben Gebietstyps (%)", ("gcn", "mlp", "gat", "gatv2"), reverse=True)
