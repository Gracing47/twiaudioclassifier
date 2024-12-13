# Trainings-Erkenntnisse: CNN vs. LSTM für Twi Audio Klassifizierung

## 1. Überblick der Trainingsexperimente

### 1.1 Modellarchitekturen
- **CNN**: Mel-Spektrogramm-basiert, 2 Conv-Layer (8->16 Filter)
- **LSTM**: MFCC-basiert, 2 BiLSTM-Layer mit Attention

### 1.2 Trainingsparameter
- Epochs: 20
- Batch Size: 16
- Learning Rate: 0.0001
- Weight Decay: 0.01
- Early Stopping (Patience: 5)

## 2. Trainingsverhalten

### 2.1 CNN-Modell
```
Epoche 1: Training Acc: 67.28% -> Val Acc: 100.00%
Epoche 5: Training Acc: 100.00% -> Val Acc: 100.00%
Finale Epoche: Training Loss: 0.0211, Val Loss: 0.0225
```

### 2.2 LSTM-Modell (Updated)
```
Epoche 1: Training Acc: 81.57% -> Val Acc: 100.00%
Epoche 5: Training Acc: 100.00% -> Val Acc: 100.00%
Finale Epoche: Training Loss: 0.0124, Val Loss: 0.0027
```

## 3. Beobachtungen und Analyse

### 3.1 Overfitting-Problematik (Updated)
- Beide Modelle zeigen weiterhin starkes Overfitting
- LSTM erreicht perfekte Validierungsgenauigkeit (100%) bereits in der ersten Epoche
- Training Loss konvergiert sehr schnell (von 0.5844 auf 0.0124)
- Sehr niedrige finale Validation Loss (0.0027) deutet auf extremes Overfitting hin

### 3.2 Vergleich der Architekturen (Updated)
1. **CNN**:
   - Langsamere initiale Konvergenz
   - Stabilere Loss-Entwicklung
   - Moderate finale Losses

2. **LSTM**:
   - Schnellere Verbesserung der Training Accuracy (81.57% -> 100%)
   - Extrem niedrige finale Losses (Training: 0.0124, Val: 0.0027)
   - Zeigt stärkeres Overfitting als das CNN-Modell

### 3.3 Probleme und Herausforderungen
1. **Datensatz**:
   - Zu kleine Validierungsmenge
   - Möglicherweise zu einfache Klassifikationsaufgabe
   - Fehlende Diversität in den Daten

2. **Modellkomplexität**:
   - Beide Modelle zu komplex für die Aufgabe
   - Zu viel Modellkapazität für die Datenmenge

## 4. Empfehlungen für Verbesserungen

### 4.1 Datenoptimierung
1. **Validierungsstrategie**:
   - Größerer Validierungssplit (20-30%)
   - K-Fold Cross-Validation
   - Stratifizierte Splits

2. **Datenerweiterung**:
   - Stärkere Augmentation
   - Synthetische Datengeneration
   - Externe Datenquellen

### 4.2 Modelloptimierung
1. **Architekturanpassungen**:
   - Weitere Reduzierung der Modellkomplexität
   - Experimentieren mit hybriden Architekturen
   - Einfachere Feature-Extraktoren

2. **Regularisierung**:
   - Stärkere L2-Regularisierung
   - Mixup Augmentation
   - Label Smoothing

### 4.3 Trainingsstrategien
1. **Hyperparameter**:
   - Höhere Learning Rate
   - Größere Batch Size
   - Aggressiveres Weight Decay

2. **Curriculum Learning**:
   - Schrittweise Erhöhung der Datenkomplexität
   - Progressive Resizing
   - Feature-basiertes Curriculum

## 5. Nächste Schritte

1. **Kurzfristig**:
   - Implementierung von K-Fold Cross-Validation
   - Weitere Reduzierung der Modellkomplexität
   - Experimentieren mit verschiedenen Feature-Extraktoren

2. **Mittelfristig**:
   - Entwicklung einer robusten Validierungsstrategie
   - Integration von Unsicherheitsschätzungen
   - Implementierung von Modell-Ensembles

3. **Langfristig**:
   - Aufbau eines größeren, diverseren Datensatzes
   - Entwicklung spezifischer Architekturen für Twi
   - Integration von Sprach-spezifischem Vorwissen

## 6. Fazit

Die aktuellen Ergebnisse zeigen, dass beide Modellarchitekturen für die gegebene Aufgabe überdimensioniert sind. Trotz verschiedener Regularisierungsansätze tritt starkes Overfitting auf. Der Fokus sollte auf der Verbesserung der Datengrundlage und der Entwicklung schlankerer Modelle liegen. Die LSTM-Architektur zeigt eine schnellere initiale Konvergenz, während das CNN-Modell stabilere Loss-Werte aufweist. Für praktische Anwendungen empfiehlt sich eine weitere Vereinfachung der Modelle und die Implementierung robusterer Evaluierungsstrategien.

## Modellarchitekturen

### CNN Modell
- **Architektur**: Reduzierte Komplexität für bessere Generalisierung
  - 2 Convolutional Layer statt 3
  - Reduzierte Filter-Anzahl (8 → 16)
  - Kleineres Dense Layer (64 Units)
- **Features**: Mel-Spektrogramme
  - Bessere Frequenzauflösung
  - Geeignet für Spracherkennung
- **Vorteile**:
  - Schnelleres Training
  - Weniger Overfitting
  - Gute Feature-Extraktion

### LSTM Modell
- **Architektur**: Bidirektionales LSTM
  - 2 Layer
  - 128 Hidden Units
  - Dropout für Regularisierung
- **Features**: MFCC
  - Kompakte Repräsentation
  - Fokus auf wichtige Frequenzbereiche
- **Vorteile**:
  - Gut für sequentielle Daten
  - Erfasst zeitliche Abhängigkeiten
  - Robust gegen Zeitverschiebungen

## Optimierungen

### Datenaugmentation
- **TimeStretch**: Moderate Streckung (0.9-1.1)
- **Masking**: 
  - Zeit- und Frequenzmasken
  - Verbesserte Robustheit
- **Rauschen**: Minimales Gaussian Noise
- **Anwendung**: 
  - CNN: Höhere Wahrscheinlichkeit (0.4)
  - LSTM: Moderate Wahrscheinlichkeit (0.3)

### Training
- **Early Stopping**:
  - CNN: Patience 7
  - LSTM: Patience 10
  - Verhindert Overfitting
- **Batch Size**:
  - CNN: 32
  - LSTM: 64
- **Learning Rate**:
  - CNN: 0.0005
  - LSTM: 0.001

## Herausforderungen

1. **Overfitting**:
   - Reduzierte Modellkomplexität
   - Verstärkte Regularisierung
   - Angepasste Augmentation

2. **Trainingszeit**:
   - Separate Konfigurationen
   - Optimierte Batch-Größen
   - Effiziente Datenverarbeitung

3. **Feature-Extraktion**:
   - Modellspezifische Features
   - Angepasste Vorverarbeitung
   - Verbesserte Augmentation

## Nächste Schritte

1. **Modell-Ensemble**:
   - Kombination beider Modelle
   - Gewichtete Vorhersagen
   - Cross-Validation

2. **Hyperparameter-Tuning**:
   - Grid Search
   - Bayesian Optimization
   - Learning Rate Scheduling

3. **Datenqualität**:
   - Erweiterte Augmentation
   - Datenbalancierung
   - Qualitätskontrolle

## Fazit

Die implementierten Optimierungen zeigen vielversprechende Ergebnisse:
- Reduzierte Modellkomplexität verhindert Overfitting
- Angepasste Augmentation verbessert Generalisierung
- Modellspezifische Konfigurationen optimieren Performance

Weitere Verbesserungen sind durch Ensemble-Methoden und systematisches Hyperparameter-Tuning möglich.
