from src.utils import read_dataset
from src.features import get_vector_mean
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Tuple
import pandas as pd
import numpy as np
import mlflow
import pickle
import os


def data_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The pipeline responsible for reading the train and test sets, applying a basic preprocessing step
    on the summary for both sets, and then splitting the training set into training and validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the training and test features array,
            respectively, followed by its labels.
    """
    path_folder = "/media/greca/HD/Datasets/Genre Classification Dataset"

    # reading the train set
    train_df = read_dataset(folder_path=path_folder, file_name="train_data.txt")

    # reading the test set
    test_df = read_dataset(folder_path=path_folder, file_name="test_data_solution.txt")

    # cleaning the summaries
    train_df["summary"] = train_df["summary"].apply(str)
    train_df["summary"] = train_df["summary"].str.lower()
    train_df["summary"] = train_df["summary"].str.strip()
    train_df["genre"] = train_df["genre"].str.strip()

    test_df["summary"] = test_df["summary"].apply(str)
    test_df["summary"] = test_df["summary"].str.lower()
    test_df["summary"] = test_df["summary"].str.strip()
    test_df["genre"] = test_df["genre"].str.strip()

    # applying the label encoder to the genre column
    le = LabelEncoder()

    train_df["genre"] = le.fit_transform(train_df["genre"])
    test_df["genre"] = le.transform(test_df["genre"])

    # saving the label encoder
    output_folder = os.path.join(os.getcwd(), "files")
    os.makedirs(output_folder, exist_ok=True)

    with open(f"{os.path.join(output_folder, 'encoder.pkl')}", mode="wb") as f:
        pickle.dump(le, f)

    # saving the preprocessed datasets
    train_df.to_csv(
        os.path.join(output_folder, "preprocessed_train.csv"), index=False, sep=","
    )
    test_df.to_csv(
        os.path.join(output_folder, "preprocessed_test.csv"), index=False, sep=","
    )

    return train_df, test_df


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Function responsible for training the Random Forest model and then
    saving it results into the current running MLflow experiment.

    Args:
        X_train (np.ndarray): the training features array.
        y_train (np.ndarray): the training labels array.
        X_test (np.ndarray): the test features array.
        y_test (np.ndarray): the test labels array.
    """
    # logging the random forest default parameters using autolog
    mlflow.sklearn.autolog()

    # training the random forest model using the default parameters
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # logging only the metrics for the test set
    prediction_test = rf.predict(X_test)
    prediction_test_proba = rf.predict_proba(X_test)

    metrics = {
        "f1 score": f1_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "recall": recall_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "precision": precision_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "auc score": roc_auc_score(
            y_true=y_test,
            y_score=prediction_test_proba,
            average="weighted",
            multi_class="ovr",
        ),
    }

    mlflow.log_metrics(metrics)

    # logging the artifacts
    mlflow.sklearn.log_model(rf, "random_forest")


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Function responsible for training the XGBoost model and then
    saving it results into the current running MLflow experiment.

    Args:
        X_train (np.ndarray): the training features array.
        y_train (np.ndarray): the training labels array.
        X_test (np.ndarray): the test features array.
        y_test (np.ndarray): the test labels array.
    """
    # logging the xgboost default parameters using autolog
    mlflow.xgboost.autolog()

    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)

    # logging only the metrics for the test set
    prediction_test = xgb.predict(X_test)
    prediction_test_proba = xgb.predict_proba(X_test)

    metrics = {
        "f1 score": f1_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "recall": recall_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "precision": precision_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "auc score": roc_auc_score(
            y_true=y_test,
            y_score=prediction_test_proba,
            average="weighted",
            multi_class="ovr",
        ),
    }

    mlflow.log_metrics(metrics)

    # logging the artifacts
    mlflow.sklearn.log_model(xgb, "xgboost")


if __name__ == "__main__":
    # creating an experiment
    experiment_id = mlflow.create_experiment(
        name="first-mlflow-experiment", tags={"version": "v1"}
    )

    # getting the data
    training_df, testing_df = data_pipeline()

    # getting the mean embedding vector of each sample
    train_X, train_y = get_vector_mean(training_df)
    test_X, test_y = get_vector_mean(testing_df)

    # creating a separate run for the Random Forest model
    with mlflow.start_run(experiment_id=experiment_id, run_name="random_forest"):
        # logging the preprocessed files
        mlflow.log_artifacts(local_dir=os.path.join(os.getcwd(), "files"))

        # logging a few of the samples used to train and test the model
        # and that will be shown as an example in the mlflow ui
        mlflow.log_input(
            mlflow.data.from_numpy(training_df["summary"].values),
            context="training_data"
        )

        mlflow.log_input(
            mlflow.data.from_numpy(testing_df["summary"].values),
            context="testing_data"
        )

        # training the random forest model and saving it
        # into the mlflow experiment run
        train_random_forest(
            X_train=train_X, y_train=train_y, X_test=test_X, y_test=test_y
        )

    # creating a separate run for the XGBoost model
    with mlflow.start_run(experiment_id=experiment_id, run_name="xgboost"):
        # logging the preprocessed files
        mlflow.log_artifacts(local_dir=os.path.join(os.getcwd(), "files"))

        # logging a few of the samples used to train and test the model
        # and that will be shown as an example in the mlflow ui
        mlflow.log_input(
            mlflow.data.from_numpy(training_df["summary"].values),
            context="training_data"
        )

        mlflow.log_input(
            mlflow.data.from_numpy(testing_df["summary"].values),
            context="testing_data"
        )

        # training the xgboost model and saving it
        # into the mlflow experiment run
        train_xgboost(
            X_train=train_X, y_train=train_y, X_test=test_X, y_test=test_y
        )
