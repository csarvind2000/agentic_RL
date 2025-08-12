from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(),
        "MLP": MLPRegressor(),
        "SVR": SVR()
    }

    param_grid = {
        "RandomForest": {"model__n_estimators": [100]},
        "MLP": {"model__hidden_layer_sizes": [(50,)], "model__alpha": [0.001]},
        "SVR": {"model__C": [1], "model__gamma": ["scale"]}
    }

    best_model = None
    best_score = float("inf")
    best_estimator = None
    best_model_name = ""

    for name, model in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        grid = GridSearchCV(pipeline, param_grid[name], cv=3)
        grid.fit(X_train, y_train)
        mse = mean_squared_error(y_test, grid.predict(X_test))
        if mse < best_score:
            best_score = mse
            best_model = grid
            best_estimator = grid.best_estimator_
            best_model_name = name

    return best_estimator, {"mse": best_score}, X_train, X_test, y_train, y_test, best_model_name

