
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


TARGET = "fuel_efficiency_mpg"
CANDIDATE_FEATURES = [
    "vehicle_weight",
    "horsepower",
    "acceleration",
    "engine_displacement",
]


def calculate_rmse(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    """Run Homework 6 Questions 2 through 6."""
    data_path = Path(__file__).with_name(
        "car_fuel_efficiency_2026.csv"
    )

    # Load data and prepare the exact train/validation split.
    df = pd.read_csv(data_path)
    df = df.fillna(0)

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )

    X_train_df = df_train.drop(columns=[TARGET])
    y_train = df_train[TARGET]

    X_val_df = df_val.drop(columns=[TARGET])
    y_val = df_val[TARGET]

    # Convert dataframes into numerical matrices.
    dv = DictVectorizer(sparse=True)

    X_train = dv.fit_transform(
        X_train_df.to_dict(orient="records")
    )
    X_val = dv.transform(
        X_val_df.to_dict(orient="records")
    )

    # QUESTION 2: Random Forest with 10 trees.
    print("\nQUESTION 2")

    rf = RandomForestRegressor(
        n_estimators=10,
        random_state=1,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    predictions = rf.predict(X_val)
    score = calculate_rmse(y_val, predictions)

    print(f"Validation RMSE: {score:.4f}")

    # QUESTION 3: Compare different numbers of trees.
    print("\nQUESTION 3")

    estimator_values = [10, 50, 100, 150]
    q3_results = {}

    for n_estimators in estimator_values:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=1,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        predictions = rf.predict(X_val)
        score = calculate_rmse(y_val, predictions)

        q3_results[n_estimators] = score

        print(
            f"Trees: {n_estimators:3d} | "
            f"RMSE: {score:.4f}"
        )

    best_n_estimators = min(
        q3_results,
        key=q3_results.get,
    )
    print(f"Best number of trees: {best_n_estimators}")

    # QUESTION 4: Compare different maximum depths.
    print("\nQUESTION 4")

    depth_values = [10, 15, 20, 25]
    depth_scores = {}

    for depth in depth_values:
        scores = []

        for n_estimators in estimator_values:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=depth,
                random_state=1,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            predictions = rf.predict(X_val)
            scores.append(
                calculate_rmse(y_val, predictions)
            )

        mean_score = float(np.mean(scores))
        depth_scores[depth] = mean_score

        print(
            f"Max depth: {depth:2d} | "
            f"Mean RMSE: {mean_score:.4f}"
        )

    best_depth = min(depth_scores, key=depth_scores.get)
    print(f"Best max_depth: {best_depth}")

    # QUESTION 5: Find the most important feature.
    print("\nQUESTION 5")

    rf = RandomForestRegressor(
        n_estimators=10,
        max_depth=20,
        random_state=1,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    encoded_features = dv.get_feature_names_out()
    feature_importances = {}

    # Aggregate one-hot feature importances by original column.
    for name, importance in zip(
        encoded_features,
        rf.feature_importances_,
    ):
        original_name = name.split("=", maxsplit=1)[0]

        feature_importances[original_name] = (
            feature_importances.get(original_name, 0.0)
            + float(importance)
        )

    missing_features = [
        feature
        for feature in CANDIDATE_FEATURES
        if feature not in feature_importances
    ]

    if missing_features:
        raise ValueError(
            f"Missing expected features: {missing_features}"
        )

    for feature in CANDIDATE_FEATURES:
        print(
            f"{feature}: "
            f"{feature_importances[feature]:.4f}"
        )

    best_feature = max(
        CANDIDATE_FEATURES,
        key=feature_importances.get,
    )
    print(f"Most important feature: {best_feature}")

    # QUESTION 6: Compare XGBoost learning rates.
    print("\nQUESTION 6")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    watchlist = [
        (dtrain, "train"),
        (dval, "validation"),
    ]

    xgb_results = {}

    for eta in [0.3, 0.1]:
        params = {
            "eta": eta,
            "max_depth": 6,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "nthread": 8,
            "seed": 1,
            "verbosity": 1,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=watchlist,
            verbose_eval=False,
        )

        predictions = model.predict(dval)
        score = calculate_rmse(y_val, predictions)

        xgb_results[eta] = score

        print(f"Eta: {eta} | Validation RMSE: {score:.4f}")

    best_eta = min(xgb_results, key=xgb_results.get)
    print(f"Best eta: {best_eta}")


if __name__ == "__main__":
    main()
