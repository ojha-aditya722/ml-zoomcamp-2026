
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def main() -> None:
    """Train a depth-one decision tree and identify its root feature."""
    data_path = Path(__file__).with_name(
        "car_fuel_efficiency_2026.csv"
    )

    df = pd.read_csv(data_path)
    df = df.fillna(0)

    # Split the dataset.
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )

    target = "fuel_efficiency_mpg"

    # Prepare training features and target.
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]

    # Convert categorical features into numerical representation.
    dv = DictVectorizer(sparse=True)
    train_dicts = X_train.to_dict(orient="records")
    X_train_encoded = dv.fit_transform(train_dicts)

    # Train the decision tree.
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X_train_encoded, y_train)

    # Identify the feature used at the root node.
    feature_index = model.tree_.feature[0]
    feature_name = dv.get_feature_names_out()[feature_index]
    root_feature = feature_name.split("=")[0]

    print("Question 1 Answer:")
    print(f"Root split feature: {root_feature}")


if __name__ == "__main__":
    main()

