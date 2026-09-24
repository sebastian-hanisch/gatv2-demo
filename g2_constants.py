"""Konstanten der GATv2-Demo: Vehikel E "Lager-Suche" (jeder Kunde sucht unter K Lagern das mit seinem Artikelcode) und Vehikel D "Liefergebiete" (wie gcn-demo, nur für falsche Kanten), Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999
LEAKY_SLOPE = 0.2

# --- Vehikel E: Lager-Suche ---------------------------------------------------------------------------------------------------------------------

N_CLASSES = 3                                   # Servicezonen (Wert eines Lagers)
CLASS_NAMES = ("Zone A", "Zone B", "Zone C")
K_MIN, K_MAX, DEFAULT_K = 2, 12, 6
CUST_MIN, CUST_MAX, CUST_STEP, DEFAULT_CUST = 100, 400, 20, 200
KNOWN_MIN, KNOWN_MAX, KNOWN_STEP, DEFAULT_KNOWN = 0.1, 0.9, 0.05, 0.75
HEADS_MIN, HEADS_MAX, DEFAULT_HEADS = 1, 8, 4
DIM_OPTIONS = (8, 16)
DEFAULT_DIM = 16

EPOCHS = 400
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
HIDDEN_GCN = 16

# --- Vehikel D: Liefergebiete ------------------------------------------------------------------------------------------------------------------

AREA = 100.0
AREA_CLASS_NAMES = ("Innenstadt", "Vorstadt", "Ländlich", "Gewerbegebiet")
AREA_N = 200
AREA_LABELS = 20
AREA_NEIGHBORS = 5
AREA_NOISE = 1.0
AREA_EPOCHS = 200
AREA_HEADS = 4
AREA_DIM = 8

# --- Experimente (feste Seeds) --------------------------------------------------------------------------------------------------------------

K_LEVELS = (2, 4, 6, 8)
HEAD_LEVELS = (1, 2, 4, 8)
EXP_CUST = 200
EXP_SEEDS_LOOKUP = tuple(range(3))
EXP_SEEDS_AREA = tuple(range(8))
WRONG_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
