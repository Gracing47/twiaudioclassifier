# Schlafqualität Vorhersage

## Projektbeschreibung
Dieses Projekt zielt darauf ab, die Schlafqualität basierend auf verschiedenen Gesundheits- und Verhaltensmetriken vorherzusagen. Wir verwenden einen synthetischen Datensatz über umfassende Schlaf- und Gesundheitsmetriken und implementieren verschiedene Machine Learning Modelle, um die beste Vorhersagegenauigkeit zu erzielen.

## 1. Datenexploration (01_datenexploration.ipynb)

In dieser ersten Phase unseres Projekts haben wir uns auf die gründliche Untersuchung und das Verständnis unseres Datensatzes konzentriert. Hier sind die wichtigsten Schritte und Erkenntnisse:

1. **Datenüberblick**: 

   Man findet den Datensatz unter "https://www.kaggle.com/datasets/uom190346a/sleep-and-health-metrics/data"
   - Wir haben die grundlegende Struktur und Statistiken des Datensatzes untersucht.
   - Der Datensatz enthält verschiedene Variablen wie Koffeinaufnahme, Stresslevel, Schlafqualität und verschiedene physiologische Messungen.

2. **Datenqualität**:
   - Wir haben nach fehlenden Werten und Duplikaten gesucht und diese behandelt.
   - Die Datentypen wurden überprüft und bei Bedarf angepasst.

3. **Explorative Datenanalyse (EDA)**:
   - Verteilungen der einzelnen Variablen wurden visualisiert und analysiert.
   - Wir haben Boxplots erstellt, um Ausreißer zu identifizieren.
   - Korrelationen zwischen den Variablen wurden mittels einer Heatmap dargestellt und analysiert.

4. **Haupterkenntnisse**:
   - Es gibt eine starke negative Korrelation zwischen Koffeinaufnahme und Schlafqualität (-0.75).
   - Stress und Koffeinaufnahme zeigen eine interessante Interaktion in Bezug auf die Schlafqualität.
   - Viele physiologische Messungen (z.B. Herzratenvariabilität, Körpertemperatur) zeigen nur schwache Korrelationen mit anderen Variablen.

5. **Identifizierte Herausforderungen**:
   - Mögliche Multikollinearität zwischen koffeinbezogenen Variablen.
   - Potenzielle nicht-lineare Beziehungen zwischen einigen Variablen.

   ## Datenbereinigung und initiales Feature Engineering

Nach der explorativen Datenanalyse wurde der Datensatz um einige grundlegende Features erweitert und als 'cleaned_sleep_health_metrics.csv' im 'data' Ordner gespeichert. Diese erweiterte Version des Datensatzes enthält sowohl die ursprünglichen Variablen als auch neu berechnete Features wie 'Sleep_Quality_Group', 'Caffeine_Stress_Interaction', und 'High_Caffeine'. Sie dient als Grundlage für weiteres Feature Engineering und die anschließende Modellierung.

Diese explorative Analyse bildet die Grundlage für unser weiteres Vorgehen im Projekt. Sie hat uns wertvolle Einblicke in die Struktur und Beziehungen innerhalb unserer Daten gegeben und wird unsere Entscheidungen in den nächsten Phasen maßgeblich beeinflussen.

## Übergang zu 02_feature_engineering.ipynb

Basierend auf den Erkenntnissen aus der Datenexploration werden wir uns im nächsten Notebook (02_feature_engineering.ipynb) auf die Verbesserung und Erweiterung unserer Features konzentrieren. Wir werden:

1. Die identifizierten Multikollinearitätsprobleme adressieren.
2. Neue Features erstellen, die die Interaktion zwischen Stress und Koffeinaufnahme besser erfassen.
3. Nicht-lineare Transformationen einiger Variablen in Betracht ziehen, um mögliche nicht-lineare Beziehungen besser zu modellieren.
4. Die physiologischen Messungen genauer untersuchen und möglicherweise aggregierte oder abgeleitete Features daraus erstellen.

Ziel ist es, einen optimierten Featureset zu erstellen, der die in der Explorationsphase gewonnenen Erkenntnisse bestmöglich nutzt und somit eine solide Grundlage für unsere anschließende Modellierung bietet.
