import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Lädt die Daten aus einer CSV-Datei.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Führt grundlegende Datenvorverarbeitung durch.
    """
    # Entfernen von Duplikaten
    df = df.drop_duplicates()
    
    # Behandlung fehlender Werte
    df = df.dropna()  # oder df.fillna(method='ffill')
    
    # Normalisierung numerischer Merkmale
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Teilt die Daten in Trainings- und Testsets.
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
