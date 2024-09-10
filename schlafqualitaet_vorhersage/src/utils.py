import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names):
    """
    Plottet die Feature-Wichtigkeit für Modelle, die feature_importances_ unterstützen.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Das Modell unterstützt keine Feature-Wichtigkeit.")

def plot_prediction_vs_actual(y_true, y_pred):
    """
    Erstellt einen Scatter-Plot der vorhergesagten vs. tatsächlichen Werte.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Tatsächliche Werte")
    plt.ylabel("Vorhergesagte Werte")
    plt.title("Vorhersage vs. Tatsächliche Werte")
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred):
    """
    Erstellt einen Residuenplot.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Vorhergesagte Werte")
    plt.ylabel("Residuen")
    plt.title("Residuenplot")
    plt.tight_layout()
    plt.show()

def save_results(results, filename):
    """
    Speichert die Ergebnisse in einer CSV-Datei.
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Ergebnisse wurden in {filename} gespeichert.")
