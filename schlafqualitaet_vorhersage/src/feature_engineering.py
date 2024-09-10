import pandas as pd
import numpy as np

def create_time_features(df, date_column):
    """
    Erstellt zeitbasierte Features aus einer Datumsspalte.
    """
    df['day_of_week'] = pd.to_datetime(df[date_column]).dt.dayofweek
    df['month'] = pd.to_datetime(df[date_column]).dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def create_lag_features(df, target_column, lag_periods=[1, 7]):
    """
    Erstellt Lag-Features für die Zielvariable.
    """
    for lag in lag_periods:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df

def create_rolling_features(df, target_column, windows=[7, 30]):
    """
    Erstellt rollende Mittelwert- und Standardabweichungs-Features.
    """
    for window in windows:
        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
        df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
    return df

def engineer_features(df, date_column, target_column):
    """
    Führt Feature-Engineering für den Datensatz durch.
    """
    df = create_time_features(df, date_column)
    df = create_lag_features(df, target_column)
    df = create_rolling_features(df, target_column)
    return df
