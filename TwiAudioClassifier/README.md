# Twi Audio Classifier

Experimenteller Ansatz zur Audio-Klassifizierung der Twi-Sprache mittels Deep Learning.

## Projektstatus und Limitationen

Dieses Projekt demonstriert die aktuellen Herausforderungen bei der automatischen Klassifizierung von Twi-Audiodaten:

### Haupterkenntnisse
- Deep Learning Ansätze (CNN, LSTM) zeigen systematische Probleme bei der Klassifizierung
- Verfügbare Datenmenge ist für robuste Modelle unzureichend
- Spezielle Charakteristika der Twi-Sprache (Tonsprache) erfordern möglicherweise angepasste Methoden

### Technische Details
- Test-Accuracy: 50% (Baseline)
- Starker Bias zu einer Klasse
- Overfitting trotz Regularisierung

Für Details siehe [Test Insights](docs/test_insights.md).

## Projektstruktur
```
TwiAudioClassifier/
├── README.md
├── TwiAudioClassifier_Arbeit.pdf  # Finale Version der Arbeit
├── config/            # Modell- und Trainingskonfigurationen
├── data/             # Datensätze
│   ├── raw/         # Original Audiodateien
│   └── processed/   # Verarbeitete Features
├── docs/            # Dokumentation und Erkenntnisse
├── experiments/     # Trainings-Experimente
├── models/         # Gespeicherte Modelle
├── notebooks/      # Jupyter Notebooks
└── src/            # Quellcode
    ├── dataloader.py   # Datenverarbeitung
    ├── models.py      # Modellarchitekturen
    ├── trainer.py     # Trainingslogik
    └── evaluate_test.py # Evaluierung
```

## Setup
1. Virtuelle Umgebung erstellen:
```bash
python -m venv venv
source venv/bin/activate  # Unix
# oder
.\venv\Scripts\activate  # Windows
```

2. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

## Modellarchitekturen

### CNN
- 2D Convolutional Neural Network
- Mel-Spektrogramm Features
- Aktuelle Performance: 50% Accuracy, starker Bias

### LSTM
- Bidirektionales LSTM
- MFCC Features
- Aktuelle Performance: 50% Accuracy, starker Bias

## Empfehlungen für zukünftige Arbeiten

1. **Datensammlung**:
   - Systematische Sammlung größerer Datensätze
   - Qualitätssicherung der Annotationen
   - Ausgewogene Klassenverteilung

2. **Alternative Ansätze**:
   - Traditional Machine Learning
   - Transfer Learning
   - Semi-supervised Learning

3. **Feature Engineering**:
   - Spezialisierte Features für Tonsprachen
   - Verbesserte Audiovorverarbeitung
   - Robustere Normalisierung

## Lizenz
MIT License

## Danksagung
Besonderer Dank an das Common Voice Projekt für die Bereitstellung der Twi-Audiodaten.
