# Test-Erkenntnisse: Modell-Bias und Klassenimbalanz

## 1. Aktuelle Probleme

### 1.1 Modell-Bias
- **CNN-Modell**:
  - Extreme Ausgabewerte: ~[10, -12] für Klasse 0/1
  - Sehr hohe Konfidenz (nahe 1.0) für Klasse 0
  - Test-Accuracy: 50% (nur durch korrekte Klasse-0-Vorhersagen)
  - Hoher Loss: ~10-17

- **LSTM-Modell**:
  - Moderatere aber immer noch biased Ausgabewerte: ~[3, -3]
  - Hohe Konfidenz (0.97-0.99) für Klasse 0
  - Test-Accuracy: 50% (nur durch korrekte Klasse-0-Vorhersagen)
  - Niedrigerer Loss: ~2.6-4.2

### 1.2 Klassenverteilung
- **Test-Set**:
  - Perfekt balanciert: 6 Samples pro Klasse
  - Alternierende Labels (0, 1, 0, 1, ...)
- **Training**:
  - Vermutete Klassenimbalanz (basierend auf Klassengewichten)
  - Klassengewichte [0.5, 2.0] deuten auf 4x mehr Klasse-0-Samples hin

### 1.3 Feature-Verarbeitung
- **Feature-Statistiken**:
  - Melspec: mean=-1.45, std=3.57
  - MFCC: mean=-0.54, std=3.18
  - Normalisierung scheint korrekt implementiert

## 2. Versuchte Lösungsansätze

### 2.1 Gewichtung der Vorhersagen
- Anwendung der Trainings-Klassengewichte [0.5, 2.0] auf:
  a) Logits direkt → verstärkte den Bias
  b) Loss-Funktion → keine Verbesserung der Vorhersagen

### 2.2 Modell-Konfiguration
- Anpassung der LSTM-Architektur an gespeicherte Gewichte:
  - Bidirektional: true
  - Hidden size: 16 (32 effektiv durch Bidirektionalität)

## 3. Hypothesen und nächste Schritte

### 3.1 Mögliche Ursachen
1. **Daten-bezogen**:
   - Signifikante Klassenimbalanz im Training
   - Systematische Unterschiede in Feature-Verteilungen
   - Potenzielle Bias in der Datensammlung

2. **Modell-bezogen**:
   - Überanpassung an dominante Klasse
   - Zu hohe Modellkapazität
   - Instabiles Training durch extreme Klassenimbalanz

### 3.2 Empfohlene Maßnahmen

1. **Datenanalyse**:
   - Klassenverteilung im Trainingsdatensatz analysieren
   - Feature-Verteilungen pro Klasse untersuchen
   - Bias in Datensammlung/Vorverarbeitung prüfen

2. **Modell-Optimierung**:
   - Modellkapazität reduzieren
   - Stärkere Regularisierung einführen (L1/L2, Dropout)
   - Batch-Normalisierung implementieren

3. **Training-Verbesserungen**:
   - Balanced Sampling statt Klassengewichte
   - Kreuzvalidierung einführen
   - Overfitting durch Monitoring verhindern

## 4. Modell-Evaluierungsmetriken

### 4.1 CNN-Modell
| Metrik    | Wert |
|-----------|------|
| Genauigkeit (Accuracy) | 50.0% |
| Präzision (Precision) | 0.50 für Klasse 0, 0.00 für Klasse 1 |
| Trefferquote (Recall) | 1.00 für Klasse 0, 0.00 für Klasse 1 |
| F1-Score | 0.67 für Klasse 0, 0.00 für Klasse 1 |
| Loss | 17.02 |

### 4.2 LSTM-Modell
| Metrik    | Wert |
|-----------|------|
| Genauigkeit (Accuracy) | 50.0% |
| Präzision (Precision) | 0.50 für Klasse 0, 0.00 für Klasse 1 |
| Trefferquote (Recall) | 1.00 für Klasse 0, 0.00 für Klasse 1 |
| F1-Score | 0.67 für Klasse 0, 0.00 für Klasse 1 |
| Loss | 3.91 |

Diese Metriken bestätigen die zuvor diskutierten Probleme mit dem Klassenungleichgewicht:
- Beide Modelle erreichen nur 50% Genauigkeit
- Sie erkennen ausschließlich Klasse 0 (perfekter Recall von 1.00)
- Keine Erkennung von Klasse 1 (Precision und F1-Score von 0.00)
- Das LSTM-Modell zeigt einen niedrigeren Loss-Wert (3.91) als das CNN-Modell (17.02)

## 5. Erwartete Verbesserungen

- Ausgewogenere Klassenvorhersagen
- Realistischere Konfidenzwerte
- Verbesserte Generalisierung
- Reduzierte Loss-Werte für beide Klassen

## 6. Fazit und Limitationen

### 6.1 Haupterkenntnisse
- Die Klassifizierung von Twi-Audiodaten stellt sich als komplexere Herausforderung dar als ursprünglich angenommen
- Beide Modellarchitekturen (CNN und LSTM) zeigen systematische Probleme bei der Klassifizierung
- Die verfügbare Datenmenge und -qualität ist für Deep Learning Ansätze suboptimal

### 6.2 Fundamentale Herausforderungen
1. **Datenakquisition**:
   - Begrenzte Verfügbarkeit von Twi-Audiodaten
   - Schwierigkeit bei der Beschaffung qualitativ hochwertiger Annotationen
   - Unausgewogene Klassenverteilung im verfügbaren Datensatz

2. **Modellkomplexität**:
   - Deep Learning Modelle benötigen typischerweise große Datenmengen
   - Aktuelle Datenmenge reicht nicht aus für robuste Generalisierung
   - Overfitting-Probleme trotz Regularisierungsmaßnahmen

3. **Sprachspezifische Aspekte**:
   - Twi als Tonsprache stellt besondere Anforderungen an die Audioanalyse
   - Komplexe phonetische und prosodische Merkmale
   - Möglicherweise ungeeignete Feature-Extraktion für Tonsprachen

### 6.3 Empfehlungen für zukünftige Projekte
1. **Alternative Ansätze**:
   - Einsatz von traditionellen ML-Methoden mit geringeren Datenanforderungen
   - Transfer Learning von verwandten Tonsprachen
   - Semi-supervised Learning zur besseren Datennutzung

2. **Datensammlung**:
   - Fokus auf systematische Datensammlung vor Modellerstellung
   - Zusammenarbeit mit Twi-Muttersprachlern für Annotationen
   - Qualitätssicherung der Audio-Aufnahmen

3. **Methodische Anpassungen**:
   - Vereinfachung der Klassifikationsaufgabe
   - Spezialisierte Feature-Extraktion für Tonsprachen
   - Robustere Evaluierungsmethoden

## Limitationen und Ausblick

### Datensatz-Limitationen

Der verwendete Mozilla Common Voice Datensatz (Version 20.0) weist einige wichtige Einschränkungen auf:

1. **Datenmenge**:
   - Nur 1 Stunde validiertes Audiomaterial
   - 11 unterschiedliche Sprecher
   - Begrenzte Variation in Aussprache und Sprechstilen

2. **Demografische Verteilung**:
   - Starke Gender-Unausgewogenheit (81% männlich)
   - Beschränkte Altersverteilung (79% zwischen 30-39 Jahre)
   - Mögliche Auswirkung auf die Generalisierbarkeit des Modells

### Verbesserungspotenziale

Für zukünftige Weiterentwicklungen des Modells empfehlen sich folgende Maßnahmen:

1. **Datenerweiterung**:
   - Sammlung zusätzlicher Twi-Audioaufnahmen
   - Fokus auf unterrepräsentierte demografische Gruppen
   - Integration verschiedener Dialekte und Sprechstile

2. **Modellanpassungen**:
   - Implementierung von Techniken zur Bias-Reduzierung
   - Verstärktes Augenmerk auf Robustheit bei verschiedenen Sprechergruppen
   - Evaluation mit diverseren Testdaten

3. **Evaluierungserweiterung**:
   - Separate Performanzanalyse für verschiedene demografische Gruppen
   - Untersuchung der Modellrobustheit bei verschiedenen Audioqualitäten
   - Vergleichstests mit anderen Sprachen

Diese Erkenntnisse sind wichtig für die realistische Einschätzung der Modellleistung und zeigen klare Wege für zukünftige Verbesserungen auf.

Diese Erkenntnisse zeigen, dass die automatische Klassifizierung von Twi-Audiodaten mit aktuellen Deep Learning Methoden noch nicht ausgereift genug ist und weitere grundlegende Forschung in diesem Bereich notwendig ist.
