from src.utils import read_dataset, bert_preprocessing, get_vector_mean
from src.bert import BERT, train_bert, test_bert
from src.dataset import create_dataloader
from torch.nn.functional import one_hot
from transformers import BertTokenizer, get_linear_schedule_with_warmup
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
import torch


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
        "test f1 score": f1_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "test recall": recall_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "test precision": precision_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "test roc auc score": roc_auc_score(
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
        "test f1 score": f1_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "test recall": recall_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "test precision": precision_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "test roc auc score": roc_auc_score(
            y_true=y_test,
            y_score=prediction_test_proba,
            average="weighted",
            multi_class="ovr",
        ),
    }

    mlflow.log_metrics(metrics)

    # logging the artifacts
    mlflow.sklearn.log_model(xgb, "xgboost")


def train_bert_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Function responsible for training the BERT model and then
    saving it results into the current running MLflow experiment.

    Args:
        train_df (np.ndarray): the training dataframe.
        test_df (np.ndarray): the training dataframe.
    """
    # defining global variables
    epochs = 7
    batch_size = 32
    max_len = 70
    lr = 2e-5

    # creating the Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # preprocessing the training and testing texts to be exactly what Bert needs
    train_ids, train_attentions = bert_preprocessing(
        texts=train_df["summary"].tolist(),
        max_len=max_len,
        tokenizer=tokenizer,
    )

    y_train = one_hot(torch.tensor(train_df["genre"].tolist()))

    test_ids, test_attentions = bert_preprocessing(
        texts=test_df["summary"].tolist(),
        max_len=max_len,
        tokenizer=tokenizer,
    )

    y_test = one_hot(torch.tensor(test_df["genre"].tolist()))

    # creating the training dataloader
    train_dataloader = create_dataloader(
        input_ids=train_ids,
        attention_masks=train_attentions,
        labels=y_train,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )

    # creating the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT(num_labels=y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, eps=1e-08)

    # creating the scheduler warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * epochs,
    )

    # creating the test dataloader
    test_dataloader = create_dataloader(
        input_ids=test_ids,
        attention_masks=test_attentions,
        labels=y_test,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )

    # logging the pytorch default parameters using autolog
    mlflow.pytorch.autolog()

    # training loop
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")
        train_metrics, train_loss = train_bert(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=train_dataloader,
            scheduler=scheduler,
        )

        test_metrics, test_loss = test_bert(
            model=model, device=device, dataloader=test_dataloader
        )

        losses = {"train loss": train_loss, "test loss": test_loss}

        # logging the metrics for each epoch
        mlflow.log_metrics(train_metrics, step=epoch)
        mlflow.log_metrics(test_metrics, step=epoch)
        mlflow.log_metrics(losses, step=epoch)

    # logging the artifacts
    mlflow.pytorch.log_model(model, "BERT")


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
            context="training_data",
        )

        mlflow.log_input(
            mlflow.data.from_numpy(testing_df["summary"].values), context="testing_data"
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
            context="training_data",
        )

        mlflow.log_input(
            mlflow.data.from_numpy(testing_df["summary"].values), context="testing_data"
        )

        # training the xgboost model and saving it
        # into the mlflow experiment run
        train_xgboost(X_train=train_X, y_train=train_y, X_test=test_X, y_test=test_y)

    # creating a separate run for the BERT model
    with mlflow.start_run(experiment_id=experiment_id, run_name="bert"):
        # logging the preprocessed files
        mlflow.log_artifacts(local_dir=os.path.join(os.getcwd(), "files"))

        # logging a few of the samples used to train and test the model
        # and that will be shown as an example in the mlflow ui
        mlflow.log_input(
            mlflow.data.from_numpy(training_df["summary"].values),
            context="training_data",
        )

        mlflow.log_input(
            mlflow.data.from_numpy(testing_df["summary"].values), context="testing_data"
        )

        # training the BERT model and saving it
        # into the mlflow experiment run
        train_bert_model(
            train_df=training_df,
            test_df=testing_df,
        )
