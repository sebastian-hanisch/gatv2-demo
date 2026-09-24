# 🔍 GATv2 – Aufmerksamkeit, die vom Fragenden abhängt

Drittes Stück der **Graph-Neural-Network-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning – und Nachfolger von
[gat-demo](https://sebastianhanisch-gat-demo.streamlit.app/) (GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer; GraphSAGE, GIN und Graph Transformer sind noch nicht gebaut).

Im Vorgänger lernte ein **GAT**, welche Nachbarn wichtig sind – aber jeder Kunde ordnet seine Nachbarn **in derselben Reihenfolge**: die Bewertung ist eine Summe aus einem Anteil des Fragenden und einem des Nachbarn
("statische Aufmerksamkeit"). **GATv2** (Brody/Alon/Yahav 2022) rechnet $a\cdot\mathrm{LeakyReLU}(W_l h_i + W_r h_j)$: die Nichtlinearität steht **vor** dem Skalarprodukt, die Rangfolge darf vom Fragenden abhängen.
Die Demo zeigt das an einer **Lager-Suche**, die genau das verlangt (jeder Kunde sucht unter seinen Lagern das mit seinem Artikelcode), und misst ehrlich, was es im Liefergebiet bei falschen Kanten bringt: nichts.
Alle Daten sind erzeugt, das Netz ist von Grund auf in numpy geschrieben (Kantenlisten, Rückwärtsrechnung von Hand).

**Bezug zu OR:** Zuordnungen wie "welches Lager bedient diesen Artikel" hängen vom Artikel ab – ein Modell, das Lager unabhängig von der Anfrage bewertet, kann das nicht abbilden.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan sah vor, dass GATv2 die im Vorgänger gemessene Schwäche behebt: GAT liegt bei **falschen Kanten** hinter dem GCN. Die **Messung hat das widerlegt**: im Liefergebiet mit zufällig ersetzten Kanten ist GATv2 nicht besser als GAT
(Unterschied in allen fünf Stufen innerhalb weniger Punkte) und liegt weiter hinter dem GCN und bei starker Störung hinter dem MLP ohne Nachbarn. Der theoretische Vorteil zeigt sich dort nicht; woran das liegt, wurde nicht untersucht.

Der Unterschied ist dennoch echt – er braucht nur eine Aufgabe, in der die Bewertung vom Fragenden abhängen **muss**. Die **Lager-Suche**: Jeder Kunde kennt einen Artikelcode $q$; seine $K$ Lager führen je genau einen Code (je Kunde neu gemischt) und liegen in
einer Servicezone; gesucht ist die Zone des Lagers, dessen Code zu $q$ passt. Die Kanten sind **gerichtet** (die Lager senden an den Kunden, der Kunde nicht an die Lager). Schlüssel und Zonen werden je Kunde neu gemischt, auswendig lernen geht nicht.
Ein GCN mittelt alle Lager (der Mittelwert ist für jede Anfrage derselbe); ein GAT ordnet die Lager für jeden Kunden gleich.

Zwei weitere Korrekturen aus der Vorab-Messung: (1) GAT scheitert nicht einfach – **mit ebenso vielen Köpfen wie Lagern** kann jeder Kopf einen Code fest bevorzugen (ein "Zeiger"), dann löst auch GAT die Suche; der Vorsprung des GATv2 entsteht erst,
wenn die Lager die Köpfe übersteigen. (2) Mehr Köpfe helfen dem GAT **nicht verlässlich**, auch wenn sie in der Theorie genügen würden (Experiment 2). Und die Suche ist **datenhungrig**: bei acht Lagern oder wenig bekannten Kunden lernt auch GATv2 sie nicht mehr sicher.

## Modell

- **Vehikel E "Lager-Suche"** (`g2_scenario.py`): je Kunde ein eigener gerichteter Stern aus Kunde und $K$ Lagern; Merkmale Anfrage-Schlüssel (nur Kunde) | Lager-Schlüssel | Zone (nur Lager), je one-hot. Standard: 6 Lager, 200 Kunden, 75 % bekannt.
- **Vehikel D "Liefergebiete"** (wie gcn-demo, nur für den Falsche-Kanten-Test): 200 Kunden, 3 Gebietstypen, 20 Etiketten je Typ, ein Anteil der Kanten zufällig ersetzt.
- **Netze** (`g2_algorithm.py`): zwei Schichten (Köpfe × Merkmale mit ReLU, dann ein Kopf mit den Klassen-Logits), Kreuzentropie auf den bekannten Knoten, Gewichtszerfall $5\cdot10^{-4}$, Adam (0,01), 400 Epochen (Lager-Suche) bzw. 200 (Liefergebiet), **kein Dropout, nichts am Test abgestimmt**.
  GCN (Mittel über die Eingangskanten bzw. $D^{-1/2}(A+I)D^{-1/2}$), GAT ($e_{ij}=\mathrm{LeakyReLU}_{0{,}2}(a_E\cdot Wh_i+a_S\cdot Wh_j)$), GATv2 (Nachricht $W_r h_j$). Alles auf **Kantenlisten** mit Segmentsummen (`np.add.reduceat`), damit die Suche mit 1 400 Knoten ohne dichte Matrix läuft.
- **Interventionstest:** man tauscht bei den Kunden die Anfrage aus und prüft, ob sich die Rangfolge der Lager nach Aufmerksamkeit (erste Schicht, je Kopf) ändert – bei GAT nie, bei GATv2 oft.

## Methodik

- **Handrechnungen:** Kantenstruktur und Segmentsummen, GCN-Gewichte gegen die dichte normierte Matrix, Softmax je Empfänger, ein Vier-Knoten-Graph, in dem bei GATv2 mit festen Gewichten die Rangfolge der zwei Lager für die zwei Fragenden **kippt** (per Hand: $e(0,A)=2{,}8$, $e(0,B)=2$; $e(1,A)=2$, $e(1,B)=2{,}8$), und ein Test, der über 40 zufällige GAT-Gewichtssätze zeigt, dass sie **nie** kippt.
- **Gegenproben:** GAT- und GATv2-Schicht gegen eine **dichte Vergleichsrechnung** (ungerichtete und gerichtete Graphen, mehrere Köpfe), **Gradienten aller Parameter gegen zentrale Differenzen** (1, 2 und 3 Köpfe, gerichtet und ungerichtet; GCN auch mit nicht symmetrischer Propagationsmatrix), Wiederholbarkeit, MLP ignoriert die Kanten.
- **Maße:** Genauigkeit auf den unbekannten Kunden, Raten = Anteil der häufigsten Zone; Trefferquote der Aufmerksamkeit (höchste Aufmerksamkeit auf dem Lager mit passendem Code); Rangfolge-Invarianz beim Austausch der Anfrage.
- **Statistik:** Experimente über feste Seeds (3 für die Lager-Suche, 8 für das Liefergebiet), Fehlerbalken = Standardfehler, Differenzen **gepaart je Seed**. Bei drei Seeds sind die Standardfehler groß.
- **Literatur** (nicht nachgebaut): Veličković et al. 2018 ("Graph attention networks", ICLR); Brody/Alon/Yahav 2022 ("How attentive are graph attention networks?", ICLR).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Standardfall (6 Lager, 4 Köpfe, Seed 3) | Ein Einzelfall auf 50 unbekannten Kunden: GCN 40,0 %, GAT 82,0 %, **GATv2 96,0 %** (Raten 38,0 %). Die Rangfolge der Lager bleibt beim GAT bei anderer Anfrage in **100 %** der Fälle gleich, beim GATv2 in 29 %. Trefferquote der Aufmerksamkeit: GAT 50 %, GATv2 80 % (Zufall 17 %). | `test_standard_preset_gatv2_solves_what_gat_and_gcn_do_not` |
| **Wie viele Lager verträgt die statische Aufmerksamkeit?** (4 Köpfe, 3 Seeds) | 2 Lager: GAT 100 %, GATv2 100 %; 4: GAT 96,0 %, GATv2 99,3 %; 6: **GAT 68,7 %, GATv2 96,0 %** (+27,3 ± 11,6 Punkte); 8: **GAT 45,3 %, GATv2 73,3 %** (+28,0 ± 6,1). GCN 46,7–67,3 % (Raten etwa 39–42 %). Bis zu so vielen Lagern wie Köpfen kommt GAT zurecht, darüber bricht es ein; auch GATv2 wird bei acht Lagern unsicher. | `test_keys_experiment_static_attention_breaks_once_depots_exceed_heads` |
| **Helfen mehr Köpfe dem GAT?** (6 Lager, 3 Seeds) | GAT: 1 Kopf 67,3 %, 2: 64,7 %, 4: 68,7 %, **8: 74,0 %** (± 11 Punkte); GATv2: 93,3 %, 96,7 %, 96,0 %, 94,7 %. Auch mit acht Köpfen (mehr als Lager) erreicht GAT die Suche **nicht verlässlich**; ein einziger GATv2-Kopf genügt. Warum das Training die festen Zeiger nicht findet, wurde nicht untersucht. | `test_heads_experiment_more_heads_do_not_reliably_rescue_gat` |
| **Falsche Kanten im Liefergebiet?** (8 Seeds) | GATv2 minus GAT: Homophilie 92 %: −0,4 ± 0,5; 81 %: +0,4 ± 1,8; 70 %: +0,5 ± 2,4; 59 %: −0,4 ± 2,6; 48 %: +2,4 ± 1,5 – **kein Unterschied**. GATv2 minus GCN: durchgehend negativ (−1,5 bis −7,2 Punkte); bei 48 % Homophilie liegt GATv2 mit 61,4 % hinter dem **MLP ohne Nachbarn (74,6 %)**. | `test_wrong_edges_gatv2_is_no_better_than_gat_and_not_better_than_gcn` |
| 2 Lager, Seed 3 | GAT und GATv2 je 100 %, GCN 70,0 %: mit zwei Lagern genügt ein fester Zeiger. | `test_two_depots_are_solved_by_both_attention_models` |
| 4 Lager (= Köpfe), Seed 3 | GAT 96,0 %, GATv2 100 %, GCN 64,0 %. | `test_as_many_depots_as_heads_gat_nearly_keeps_up` |
| 8 Lager, Seed 3 | GCN 46,0 %, GAT 54,0 %, GATv2 80,0 % (Raten 42,0 %). | `test_eight_depots_are_hard_but_gatv2_still_leads` |
| 1 Kopf, Seed 3 | GAT 66,0 %, GATv2 90,0 %, GCN 40,0 %; die Rangfolge der Lager ändert sich beim GATv2 mit der Anfrage in allen Fällen (0 % gleich). | `test_one_head_gatv2_still_finds_the_match_while_the_ranking_always_changes` |
| Wenige bekannte Kunden (30 %), Seed 3 | GCN 39,3 %, GAT 48,6 %, GATv2 56,4 % (Raten 37,1 %): zu wenig, um "passt" zu lernen – dynamische Aufmerksamkeit ist datenhungrig. | `test_few_known_customers_make_it_data_hungry` |

Die Preset-Zeilen sind **Einzelfälle** (Seed 3; die Zahlen wandern von Seed zu Seed, etwa liegt GAT im Standardfall im Mittel über drei Seeds bei 68,7 % statt 82,0 %); die Tests prüfen dort nur Strukturgrenzen. Belastbar sind die Mehr-Seed-Zeilen.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Beispiele, um "passt" zu lernen** | GATv2 ist ausdrucksstärker, aber datenhungrig: mit wenigen bekannten Kunden oder vielen Lagern lernen auch GATv2 und GAT die Suche nicht zuverlässig. | – |
| **Die Aufgabe verlangt Abhängigkeit vom Fragenden** | Bei Homophilie-Aufgaben reicht statische Aufmerksamkeit oder ein Mittel; im Liefergebiet mit falschen Kanten bringt GATv2 nichts. | – |
| **Der Fragende sieht den Nachbarn nicht anders als andere Fragende** | Bedeutet ein Nachbar nur in Kombination etwas, hilft Wechselwirkung (GATv2, Transformer) oder viele Köpfe als feste Zeiger. | GATv2 (hier) |
| **Der ganze Graph ist beim Training bekannt** | Wie GCN und GAT transduktiv; neue Kunden oder sehr große Graphen brauchen Stichproben der Nachbarschaft. Hier nicht gemessen. | GraphSAGE |
| **Erzeugte Daten, wenige Seeds** | Die Lager-Suche ist ein synthetischer Prüfstein, drei Seeds je Messpunkt, große Standardfehler; ReLU statt ELU, kein Dropout. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund 6–7 Minuten wegen der Experimente): Kern per Handrechnung und Gegenprobe (Kantenstruktur, Softmax, GCN-Gewichte, GAT/GATv2 gegen dichte Rechnung, Rangfolge kippt nur bei GATv2, Gradienten), Vehikel, Auswertung
(Aufmerksamkeit je Lager von Hand, Interventionstest, Experimentstruktur), Preset- und Permalink-Klemmen, AppTest-Rauchtests (jedes Preset, Kundenwahl, Extremwerte, drei Experimente auf Abruf) und `test_claims.py` (jede Zahl aus diesem README; Einzelfälle nur mit Strukturgrenzen, Mehr-Seed-Zahlen mit großzügigen Bändern).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `g2_constants.py` | Regler-Grenzen, Netz-Konstanten, Experiment-Seeds |
| `g2_presets.py` | Permalink/Presets-Mechanik |
| `g2_scenario.py` | Vehikel E (Lager-Suche) und D (Liefergebiete) |
| `g2_algorithm.py` | Kantenlisten, GCN/GAT/GATv2 (Schichten, Gradienten, Adam) |
| `g2_evaluation.py` | Analyse, Aufmerksamkeitsmaße, Interventionstest, drei Experimente |
| `g2_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Dropout, ELU, Early Stopping, Feinabstimmung der Hyperparameter, mehr als acht Köpfe, mehr als acht Lager.
- Die Ursachenforschung, warum GATv2 bei falschen Kanten nicht hilft und warum GAT mit acht Köpfen die Zeiger nicht sicher findet.
- Die übrigen Stücke der Linie (GraphSAGE, GIN, Graph Transformer); ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
