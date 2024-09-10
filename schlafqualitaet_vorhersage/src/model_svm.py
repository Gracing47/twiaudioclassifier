from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_svm_model(X_train, y_train):
    """
    Trainiert ein SVM-Modell mit Hyperparameter-Tuning.
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }
    
    svm = SVR()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_svm = grid_search.best_estimator_
    return best_svm

def evaluate_svm_model(model, X_test, y_test):
    """
    Evaluiert das SVM-Modell auf den Testdaten.
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
