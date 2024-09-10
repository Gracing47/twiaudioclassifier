from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def create_ensemble_model(models):
    """
    Erstellt ein Ensemble-Modell aus den gegebenen Modellen.
    """
    ensemble = VotingRegressor(estimators=models)
    return ensemble

def train_ensemble_model(ensemble, X_train, y_train):
    """
    Trainiert das Ensemble-Modell.
    """
    ensemble.fit(X_train, y_train)
    return ensemble

def evaluate_ensemble_model(model, X_test, y_test):
    """
    Evaluiert das Ensemble-Modell auf den Testdaten.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
