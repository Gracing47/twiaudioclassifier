# Twi Audio Dataset Analyse

*Hinweis: Diese Analyse basiert auf einem ausgewählten Subset des Mozilla Common Voice Datensatzes (Version 20.0), der für dieses Projekt speziell aufbereitet wurde.*

## 1. Datensatz-Übersicht

### 1.1 Größe und Aufteilung
- **Gesamtanzahl Clips**: 190
- **Training Set**: 178 Clips (93.7%)
- **Test Set**: 12 Clips (6.3%)
- **Gesamtdauer**: 0.33 Stunden (~20 Minuten)

### 1.2 Audio-Eigenschaften
- **Durchschnittliche Dauer**: 4.31 Sekunden
- **Standardabweichung**: 1.75 Sekunden
- **Sample Rate**: 16 kHz
- **Format**: MP3

### 1.3 Sprecherdemographie
- **Geschlecht**: 100% männlich
- **Altersgruppe**: 100% in den Dreißigern
- **Sprachkompetenz**: 100% Muttersprachler (Native)

## 2. Qualitätsanalyse

### 2.1 Audio-Qualität
- **Validierte Clips**: 178/178 (100%)
- **Ungültige Clips**: 0
- **Qualitätskriterien**:
  - Minimale Dauer: 1.0 Sekunden
  - Maximale Dauer: 10.0 Sekunden
  - Ausreichende Signalstärke
  - Keine NaN-Werte oder Artefakte

### 2.2 Datenverteilung
Die Audiodaten zeigen eine relativ gleichmäßige Verteilung der Dauern, was für das Training vorteilhaft ist.

## 3. Visualisierungen

Die folgenden Visualisierungen wurden erstellt und sind im `/docs/plots` Verzeichnis verfügbar:

1. **Duration Distribution** (`duration_dist.png`)
   - Zeigt die Verteilung der Audio-Dauern
   - Hauptsächlich zwischen 2-7 Sekunden
   - Normalverteilung mit leichter Rechtsschiefe

2. **Gender Distribution** (`gender_dist.png`)
   - Zeigt die Geschlechterverteilung
   - Ausschließlich männliche Sprecher

3. **Age Distribution** (`age_dist.png`)
   - Zeigt die Altersverteilung
   - Konzentriert in der Altersgruppe "Dreißiger"

## 4. Implikationen für das Training

### 4.1 Vorteile
- Konsistente Audioqualität
- Einheitliche Sprechercharakteristiken
- Gut validierte Daten

### 4.2 Herausforderungen
- Begrenzte Datenmenge (20 Minuten)
- Keine Geschlechterdiversität
- Eingeschränkte Altersverteilung

### 4.3 Empfehlungen
1. **Datenaugmentation**:
   - Pitch Shifting
   - Time Stretching
   - Noise Injection
   
2. **Trainingsstrategien**:
   - Kreuzvalidierung wegen begrenzter Datenmenge
   - Frühe Stopping zur Vermeidung von Overfitting
   - Regularisierung wegen kleinem Dataset

3. **Modellanpassungen**:
   - Eher kleinere Modellarchitekturen
   - Dropout für bessere Generalisierung
   - Batch Normalization für stabileres Training

## 5. Nächste Schritte

1. **Feature-Extraktion**:
   - MFCC Features für LSTM
   - Mel-Spektrogramme für CNN
   
2. **Datenaugmentation**:
   - Implementation der empfohlenen Augmentierungstechniken
   - Validierung der augmentierten Daten

3. **Modellierung**:
   - Training des CNN-Modells
   - Training des LSTM-Modells
   - Vergleich der Performanz

## 6. Zusätzliche Bemerkungen

Der Datensatz, obwohl klein, bietet eine solide Grundlage für ein Proof-of-Concept der Twi-Sprachklassifizierung. Die hohe Qualität und Konsistenz der Aufnahmen sollte zu stabilen Trainingsergebnissen führen, vorausgesetzt, geeignete Techniken zur Vermeidung von Overfitting werden eingesetzt.

## 7. Feature Extraktion

### 7.1 Implementierte Features

#### MFCC (für LSTM)
- **Dimensionen**: [157, 40]
  - 157 Zeitschritte
  - 40 MFCC Koeffizienten pro Zeitschritt
- **Parameter**:
  - Sample Rate: 16kHz
  - FFT-Größe: 2048
  - Hop Length: 512
  - Mel-Bänder: 128
- **Normalisierung**:
  - Standardisierung (mean=-4.27, std=4.57)
  - Längen-Normalisierung auf 5 Sekunden

#### Mel-Spektrogramm (für CNN)
- **Dimensionen**: [1, 1, 128, 157]
  - 1 Batch
  - 1 Kanal
  - 128 Mel-Frequenzbänder
  - 157 Zeitschritte
- **Parameter**:
  - Sample Rate: 16kHz
  - FFT-Größe: 2048
  - Hop Length: 512
  - Power: 2.0
- **Verarbeitung**:
  - Log-Skalierung
  - Per-Feature Normalisierung

### 7.2 Vorverarbeitungspipeline

1. **Audio Laden**:
   - Resampling auf 16kHz
   - Konvertierung zu Mono
   - Padding/Truncating auf 5 Sekunden

2. **Feature Berechnung**:
   - MFCC Extraktion für LSTM
   - Mel-Spektrogramm Berechnung für CNN
   - Normalisierung beider Features

3. **Qualitätssicherung**:
   - Validierung der Audio-Länge
   - Prüfung auf NaN-Werte
   - Signalstärke-Überprüfung

### 7.3 Technische Details

```python
# Feature Konfiguration
config = FeatureConfig(
    sample_rate=16000,
    n_mfcc=40,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    target_length=5.0
)

# Beispiel Feature Shapes
MFCC (LSTM): torch.Size([157, 40])
Mel-Spec (CNN): torch.Size([1, 1, 128, 157])
```

### 7.4 Speicheranforderungen

- **MFCC Features**:
  - ~25KB pro Audio-Clip
  - ~4.5MB für gesamten Datensatz

- **Mel-Spektrogramme**:
  - ~80KB pro Audio-Clip
  - ~15MB für gesamten Datensatz

### 7.5 Nächste Schritte

1. **DataLoader Implementation**:
   - Batch-Verarbeitung
   - Shuffling
   - Augmentation im Datenpipeline

2. **Feature Optimierung**:
   - Hyperparameter-Tuning der Feature-Extraktion
   - Experimentieren mit verschiedenen Normalisierungstechniken
   - Implementierung von Online-Augmentation
