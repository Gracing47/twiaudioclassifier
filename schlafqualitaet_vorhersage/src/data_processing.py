import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Lädt die Rohdaten aus einer CSV-Datei.
    
    :param filepath: Pfad zur CSV-Datei
    :return: pandas DataFrame mit den geladenen Daten
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Führt grundlegende Vorverarbeitungsschritte für die Daten durch.
    
    :param df: pandas DataFrame mit den Rohdaten
    :return: Vorverarbeiteter pandas DataFrame
    """
    # Hier Vorverarbeitungsschritte einfügen, z.B.:
    # - Behandlung fehlender Werte
    # - Kodierung kategorischer Variablen
    # - Normalisierung numerischer Variablen
    
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Teilt die Daten in Trainings- und Testsets.
    
    :param df: Vorverarbeiteter pandas DataFrame
    :param target_column: Name der Zielvariable
    :param test_size: Anteil der Daten für das Testset
    :param random_state: Seed für die Zufallsgenerierung
    :return: X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Beispielverwendung
    data = load_data("../data/raw/sleep_data.csv")
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data, "sleep_quality")
    print("Daten erfolgreich geladen und verarbeitet.")
