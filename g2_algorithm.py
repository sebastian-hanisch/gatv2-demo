"""GCN, GAT (Veličković et al. 2018) und GATv2 (Brody/Alon/Yahav 2022) auf Kantenlisten, von Grund auf in numpy.

Alle Netze haben zwei Schichten (Köpfe x Merkmale mit ReLU, dann Klassen-Logits), Softmax-Kreuzentropie auf den bekannten Knoten, Gewichtszerfall auf den Matrizen, Adam, feste Epochenzahl, kein Dropout.
Die Bewertung einer Kante j -> i ist
- GAT:   e_ij = LeakyReLU( a_E . Wh_i + a_S . Wh_j )       (Nichtlinearität NACH der Summe: "statische" Aufmerksamkeit - die Rangfolge der Nachbarn hängt nur von j ab)
- GATv2: e_ij = a . LeakyReLU( W_l h_i + W_r h_j )         (Nichtlinearität VOR dem Skalarprodukt: die Rangfolge kann von i abhängen - "dynamische" Aufmerksamkeit)
Kanten werden als Liste (Empfänger, Sender) gehalten, sortiert nach Empfänger; jeder Knoten hat eine Schleife."""

from dataclasses import dataclass, field

import numpy as np

import g2_constants as C


def leaky(z):
    return np.where(z > 0, z, C.LEAKY_SLOPE * z)


def dleaky(z):
    return np.where(z > 0, 1.0, C.LEAKY_SLOPE)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


@dataclass
class Edges:
    n: int
    dst: np.ndarray
    src: np.ndarray
    ptr_dst: np.ndarray
    order_src: np.ndarray
    ptr_src: np.ndarray

    def seg_dst(self, v):
        """Summe je Empfänger (Kanten sind nach Empfänger sortiert)."""
        return np.add.reduceat(v, self.ptr_dst, axis=0)

    def seg_src(self, v):
        """Summe je Sender."""
        return np.add.reduceat(v[self.order_src], self.ptr_src, axis=0)

    @property
    def n_edges(self):
        return len(self.dst)


def edges_of(A):
    """Kantenliste aus A + I (Zeile i = Sender j -> Empfänger i)."""
    n = len(A)
    dst, src = np.nonzero((A + np.eye(n)) > 0)
    order = np.argsort(src, kind="stable")
    ptr_dst = np.searchsorted(dst, np.arange(n))
    ptr_src = np.searchsorted(src[order], np.arange(n))
    return Edges(n, dst, src, ptr_dst, order, ptr_src)


def edge_softmax(e, g):
    """Softmax über die eingehenden Kanten jedes Empfängers (e: (E,) oder (E, K))."""
    m = np.maximum.reduceat(e, g.ptr_dst, axis=0)
    ex = np.exp(e - m[g.dst])
    return ex / g.seg_dst(ex)[g.dst]


def _glorot(rng, a, b):
    lim = np.sqrt(6.0 / (a + b))
    return rng.uniform(-lim, lim, size=(a, b))


def _adam(params, grads, mo, ve, t, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in params:
        mo[k] = b1 * mo[k] + (1 - b1) * grads[k]
        ve[k] = b2 * ve[k] + (1 - b2) * grads[k] ** 2
        params[k] = params[k] - lr * (mo[k] / (1 - b1 ** t)) / (np.sqrt(ve[k] / (1 - b2 ** t)) + eps)


# --- GCN / MLP ------------------------------------------------------------------------------------------------------------------------------------


def gcn_weights(g, mode="sym"):
    """Kantengewichte: 'sym' = 1/sqrt(d_i d_j) (ungerichtete Graphen, wie A_hat = D^-1/2 (A+I) D^-1/2), 'mean' = 1/d_i (Mittel über die Eingangskanten, für gerichtete), 'self' = nur Schleifen (MLP)."""
    deg = g.seg_dst(np.ones(g.n_edges))
    if mode == "sym":
        return 1.0 / np.sqrt(deg[g.dst] * deg[g.src])
    if mode == "mean":
        return 1.0 / deg[g.dst]
    return (g.dst == g.src).astype(float)


def gcn_forward(g, w, X, p):
    AH1 = g.seg_dst(w[:, None] * X[g.src])
    Z1 = AH1 @ p["W1"]
    H1 = np.maximum(Z1, 0.0)
    AH2 = g.seg_dst(w[:, None] * H1[g.src])
    return AH2 @ p["W2"], (AH1, Z1, H1, AH2)


def gcn_loss_and_grads(g, w, X, y, train, p, wd):
    logits, (AH1, Z1, H1, AH2) = gcn_forward(g, w, X, p)
    P = softmax(logits)
    idx = np.flatnonzero(train)
    ce = -np.log(P[idx, y[idx]] + C.EPS).mean()
    reg = 0.5 * wd * sum(float((p[k] ** 2).sum()) for k in ("W1", "W2"))
    dZ2 = np.zeros_like(logits)
    dZ2[idx] = P[idx]
    dZ2[idx, y[idx]] -= 1.0
    dZ2 /= len(idx)
    dW2 = AH2.T @ dZ2
    dH1 = g.seg_src(w[:, None] * (dZ2 @ p["W2"].T)[g.dst])          # transponierte Propagation
    dZ1 = dH1 * (Z1 > 0)
    dW1 = AH1.T @ dZ1
    return ce + reg, {"W1": dW1 + wd * p["W1"], "W2": dW2 + wd * p["W2"]}


# --- GAT-Schicht ----------------------------------------------------------------------------------------------------------------------------------


def gat_layer(H, W, a_src, a_dst, g):
    heads, f = a_src.shape
    Whh = (H @ W).reshape(g.n, heads, f)
    s = np.einsum("nhf,hf->nh", Whh, a_src)
    t = np.einsum("nhf,hf->nh", Whh, a_dst)
    z = s[g.dst] + t[g.src]
    al = edge_softmax(leaky(z), g)
    out = g.seg_dst(al[:, :, None] * Whh[g.src])
    return out.reshape(g.n, heads * f), (H, Whh, z, al)


def gat_layer_back(dout, W, a_src, a_dst, g, cache):
    H, Whh, z, al = cache
    heads, f = a_src.shape
    dO = dout.reshape(g.n, heads, f)
    dal = np.einsum("ehf,ehf->eh", dO[g.dst], Whh[g.src])
    dWhh = g.seg_src(al[:, :, None] * dO[g.dst])
    tot = g.seg_dst(al * dal)
    de = al * (dal - tot[g.dst])
    dz = de * dleaky(z)
    ds, dt = g.seg_dst(dz), g.seg_src(dz)
    dWhh += ds[:, :, None] * a_src[None] + dt[:, :, None] * a_dst[None]
    dWh = dWhh.reshape(g.n, heads * f)
    return dWh @ W.T, H.T @ dWh, np.einsum("nhf,nh->hf", Whh, ds), np.einsum("nhf,nh->hf", Whh, dt)


# --- GATv2-Schicht --------------------------------------------------------------------------------------------------------------------------------


def gatv2_layer(H, Wl, Wr, a, g):
    heads, f = a.shape
    L = (H @ Wl).reshape(g.n, heads, f)
    R = (H @ Wr).reshape(g.n, heads, f)
    U = L[g.dst] + R[g.src]
    Z = leaky(U)
    e = np.einsum("ehf,hf->eh", Z, a)
    al = edge_softmax(e, g)
    out = g.seg_dst(al[:, :, None] * R[g.src])
    return out.reshape(g.n, heads * f), (H, L, R, U, Z, al)


def gatv2_layer_back(dout, Wl, Wr, a, g, cache):
    H, L, R, U, Z, al = cache
    heads, f = a.shape
    dO = dout.reshape(g.n, heads, f)
    dal = np.einsum("ehf,ehf->eh", dO[g.dst], R[g.src])
    dR = g.seg_src(al[:, :, None] * dO[g.dst])
    tot = g.seg_dst(al * dal)
    de = al * (dal - tot[g.dst])
    da = np.einsum("eh,ehf->hf", de, Z)
    dU = de[:, :, None] * a[None] * dleaky(U)
    dL = g.seg_dst(dU)
    dR += g.seg_src(dU)
    dLf, dRf = dL.reshape(g.n, heads * f), dR.reshape(g.n, heads * f)
    return dLf @ Wl.T + dRf @ Wr.T, H.T @ dLf, H.T @ dRf, da


# --- Modelle --------------------------------------------------------------------------------------------------------------------------------------


def init_params(kind, d, heads, f, K, seed, hidden=C.HIDDEN_GCN):
    rng = np.random.default_rng([seed, 11])
    if kind in ("gcn", "mlp"):
        return {"W1": _glorot(rng, d, hidden), "W2": _glorot(rng, hidden, K)}
    if kind == "gat":
        return {"W1": _glorot(rng, d, heads * f), "as1": rng.uniform(-0.3, 0.3, (heads, f)), "ad1": rng.uniform(-0.3, 0.3, (heads, f)),
                "W2": _glorot(rng, heads * f, K), "as2": rng.uniform(-0.3, 0.3, (1, K)), "ad2": rng.uniform(-0.3, 0.3, (1, K))}
    return {"Wl1": _glorot(rng, d, heads * f), "Wr1": _glorot(rng, d, heads * f), "a1": rng.uniform(-0.3, 0.3, (heads, f)),
            "Wl2": _glorot(rng, heads * f, K), "Wr2": _glorot(rng, heads * f, K), "a2": rng.uniform(-0.3, 0.3, (1, K))}


def forward(kind, p, X, g, w=None):
    """Logits und Cache; w = Kantengewichte für gcn/mlp."""
    if kind in ("gcn", "mlp"):
        return gcn_forward(g, w, X, p)
    if kind == "gat":
        o1, c1 = gat_layer(X, p["W1"], p["as1"], p["ad1"], g)
        o2, c2 = gat_layer(np.maximum(o1, 0.0), p["W2"], p["as2"], p["ad2"], g)
    else:
        o1, c1 = gatv2_layer(X, p["Wl1"], p["Wr1"], p["a1"], g)
        o2, c2 = gatv2_layer(np.maximum(o1, 0.0), p["Wl2"], p["Wr2"], p["a2"], g)
    return o2, (o1, c1, c2)


def loss_and_grads(kind, p, X, y, train, g, wd, w=None):
    if kind in ("gcn", "mlp"):
        return gcn_loss_and_grads(g, w, X, y, train, p, wd)
    logits, (o1, c1, c2) = forward(kind, p, X, g)
    P = softmax(logits)
    idx = np.flatnonzero(train)
    ce = -np.log(P[idx, y[idx]] + C.EPS).mean()
    wk = ("W1", "W2") if kind == "gat" else ("Wl1", "Wr1", "Wl2", "Wr2")
    reg = 0.5 * wd * sum(float((p[k] ** 2).sum()) for k in wk)
    dl = np.zeros_like(logits)
    dl[idx] = P[idx]
    dl[idx, y[idx]] -= 1.0
    dl /= len(idx)
    if kind == "gat":
        dh1, dW2, das2, dad2 = gat_layer_back(dl, p["W2"], p["as2"], p["ad2"], g, c2)
        _, dW1, das1, dad1 = gat_layer_back(dh1 * (o1 > 0), p["W1"], p["as1"], p["ad1"], g, c1)
        grads = {"W1": dW1 + wd * p["W1"], "as1": das1, "ad1": dad1, "W2": dW2 + wd * p["W2"], "as2": das2, "ad2": dad2}
    else:
        dh1, dWl2, dWr2, da2 = gatv2_layer_back(dl, p["Wl2"], p["Wr2"], p["a2"], g, c2)
        _, dWl1, dWr1, da1 = gatv2_layer_back(dh1 * (o1 > 0), p["Wl1"], p["Wr1"], p["a1"], g, c1)
        grads = {"Wl1": dWl1 + wd * p["Wl1"], "Wr1": dWr1 + wd * p["Wr1"], "a1": da1, "Wl2": dWl2 + wd * p["Wl2"], "Wr2": dWr2 + wd * p["Wr2"], "a2": da2}
    return ce + reg, grads


@dataclass
class Model:
    kind: str
    params: dict
    w: object = None
    history: dict = field(default_factory=dict)


def train(kind, A, X, y, train_mask, test_mask, seed=0, heads=C.DEFAULT_HEADS, f=C.DEFAULT_DIM, epochs=C.EPOCHS, lr=C.LEARNING_RATE, wd=C.WEIGHT_DECAY, gcn_mode="sym", record_pred=False):
    """kind: 'gcn', 'mlp', 'gat', 'gatv2'. Test = Knoten der test_mask (nur zur Anzeige, nie zur Auswahl)."""
    g = edges_of(A)
    K = int(y.max()) + 1
    w = gcn_weights(g, "self" if kind == "mlp" else gcn_mode) if kind in ("gcn", "mlp") else None
    p = init_params(kind, X.shape[1], heads, f, K, seed)
    mo = {k: np.zeros_like(v) for k, v in p.items()}
    ve = {k: np.zeros_like(v) for k, v in p.items()}
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    preds = []
    for t in range(1, epochs + 1):
        loss, grads = loss_and_grads(kind, p, X, y, train_mask, g, wd, w)
        _adam(p, grads, mo, ve, t, lr)
        pred = forward(kind, p, X, g, w)[0].argmax(axis=1)
        hist["loss"].append(float(loss))
        hist["train_acc"].append(float((pred[train_mask] == y[train_mask]).mean()))
        hist["test_acc"].append(float((pred[test_mask] == y[test_mask]).mean()))
        if record_pred:
            preds.append(pred)
    if record_pred:
        hist["pred"] = np.array(preds)
    return Model(kind, p, w, hist)


def predict(model, A, X):
    g = edges_of(A)
    return forward(model.kind, model.params, X, g, model.w)[0].argmax(axis=1)


def attention_layer1(model, A, X):
    """Aufmerksamkeit der ersten Schicht je Kante und Kopf: (alpha (E, Köpfe), Kantenliste)."""
    g = edges_of(A)
    if model.kind == "gat":
        _, (o1, c1, c2) = forward("gat", model.params, X, g)
        return c1[3], g
    _, (o1, c1, c2) = forward("gatv2", model.params, X, g)
    return c1[5], g
