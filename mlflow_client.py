from src.utils import bert_preprocessing, get_vector_mean
from src.bert import BERT, train_bert, test_bert
from src.dataset import create_dataloader
from tracking import data_pipeline
from torch.nn.functional import one_hot
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from xgboost import XGBClassifier
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from typing import Union, Tuple
import torch
import argparse
import mlflow
import pandas as pd
import numpy as np
import pickle
import os


def calculate_metrics(
    model: Union[RandomForestClassifier, XGBClassifier],
    y_true: np.ndarray,
    X: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Function responsible for assessing the trained model with a given set.

    Args:
        model (Union[RandomForestClassifier, XGBClassifier]): the trained model
            (Random Forest Classifier or XGBoost Classifier).
        y_true (np.ndarray): the labels ground-truth.
        X (np.ndarray): the set features.

    Returns:
        Tuple[float, float, float, float]: the f1 score, recall, precision, and
            roc auc scores, respectively.
    """
    # logging only the metrics for the test set
    prediction_validation = model.predict(X)
    prediction_validation_proba = model.predict_proba(X)

    # calculating the metrics
    f1 = f1_score(
        y_true=y_true,
        y_pred=prediction_validation,
        average="weighted",
        zero_division=0.0,
    )

    recall = recall_score(
        y_true=y_true,
        y_pred=prediction_validation,
        average="weighted",
        zero_division=0.0,
    )

    precision = precision_score(
        y_true=y_true,
        y_pred=prediction_validation,
        average="weighted",
        zero_division=0.0,
    )

    roc_auc = roc_auc_score(
        y_true=y_true,
        y_score=prediction_validation_proba,
        average="weighted",
        multi_class="ovr",
    )

    return f1, recall, precision, roc_auc


def train_bert_model(
    client: MlflowClient,
    run: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Function responsible for training the BERT model and then
    saving its metrics and artifacts into the mlflow run.

    Args:
        client (MlflowClient): the MLflow Client object.
        run (str): the run id.
        train_df (pd.DataFrame): the training dataframe.
        valid_df (pd.DataFrame): the validation dataframe.
        test_df (pd.DataFrame): the testing dataframe.
    """
    # defining global variables
    epochs = 4
    batch_size = 32
    max_len = 70
    lr = 2e-5

    # creating the Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # saving the tokenizer
    with open(f"{os.path.join('.', 'files', 'bert_tokenizer.pkl')}", mode="wb") as f:
        pickle.dump(tokenizer, f)

    # preprocessing the training, validation, and testing texts to be exactly what Bert needs
    train_ids, train_attentions = bert_preprocessing(
        texts=train_df["summary"].tolist(),
        max_len=max_len,
        tokenizer=tokenizer,
    )

    y_train = one_hot(torch.tensor(train_df["genre"].tolist()))

    valid_ids, valid_attentions = bert_preprocessing(
        texts=valid_df["summary"].tolist(),
        max_len=max_len,
        tokenizer=tokenizer,
    )

    y_valid = one_hot(torch.tensor(valid_df["genre"].tolist()))

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

    # creating the validation dataloader
    valid_dataloader = create_dataloader(
        input_ids=valid_ids,
        attention_masks=valid_attentions,
        labels=y_valid,
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

    # training loop
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")
        _, _ = train_bert(
            model=model,
            optimizer=optimizer,
            device=device,
            dataloader=train_dataloader,
            scheduler=scheduler,
        )

        valid_metrics, valid_loss = test_bert(
            model=model, device=device, dataloader=valid_dataloader
        )

    # saving the trained model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": valid_loss,
        },
        os.path.join(".", "files", "bert.pt"),
    )

    # logging the artifacts
    client.log_artifact(run, "files/bert.pt")
    client.log_artifact(run, "files/bert_tokenizer.pkl")
    client.log_artifact(run, "files/preprocessed_test.csv")
    client.log_artifact(run, "files/preprocessed_train.csv")

    # logging the metrics
    client.log_metric(run, "validation_f1_score", valid_metrics["validation f1 score"])
    client.log_metric(run, "validation_recall", valid_metrics["validation recall"])
    client.log_metric(
        run, "validation_precision", valid_metrics["validation precision"]
    )

    test_metrics, _ = test_bert(model=model, device=device, dataloader=test_dataloader)

    client.log_metric(run, "test_f1_score", test_metrics["validation f1 score"])
    client.log_metric(run, "test_recall", test_metrics["validation recall"])
    client.log_metric(run, "test_precision", test_metrics["validation precision"])


def train_xgboost(
    client: MlflowClient,
    run: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Function responsible for training the XGBoost model and then
    saving its metrics and artifacts into the mlflow run.

    Args:
        client (MlflowClient): the MLflow Client object.
        run (str): the run id.
        X_train (np.ndarray): the training features.
        y_train (np.ndarray): the training labels.
        X_valid (np.ndarray): the validation features.
        y_valid (np.ndarray): the validation labels.
        X_test (np.ndarray): the testing features.
        y_test (np.ndarray): the testing labels.
    """
    # training the xgboost model using the default parameters
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)

    # saving the trained model
    with open(f"{os.path.join('.', 'files', 'xgboost.pkl')}", mode="wb") as f:
        pickle.dump(xgb, f)

    # logging the artifacts
    client.log_artifact(run, "files/encoder.pkl")
    client.log_artifact(run, "files/xgboost.pkl")
    client.log_artifact(run, "files/preprocessed_test.csv")
    client.log_artifact(run, "files/preprocessed_train.csv")

    # calculating the metrics for the validation set
    valid_f1, valid_recall, valid_precision, valid_roc_auc = calculate_metrics(
        model=xgb, y_true=y_valid, X=X_valid
    )

    # logging the metrics
    client.log_metric(run, "validation_f1_score", valid_f1)
    client.log_metric(run, "validation_recall", valid_recall)
    client.log_metric(run, "validation_precision", valid_precision)
    client.log_metric(run, "validation_roc_auc_score", valid_roc_auc)

    # calculating the metrics for the test set
    test_f1, test_recall, test_precision, test_roc_auc = calculate_metrics(
        model=xgb, y_true=y_test, X=X_test
    )

    # logging the metrics
    client.log_metric(run, "test_f1_score", test_f1)
    client.log_metric(run, "test_recall", test_recall)
    client.log_metric(run, "test_precision", test_precision)
    client.log_metric(run, "test_roc_auc_score", test_roc_auc)


def train_random_forest(
    client: MlflowClient,
    run: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Function responsible for training the Random Forest model and then
    saving its metrics and artifacts into the mlflow run.

    Args:
        client (MlflowClient): the MLflow Client object.
        run (str): the run id.
        X_train (np.ndarray): the training features.
        y_train (np.ndarray): the training labels.
        X_valid (np.ndarray): the validation features.
        y_valid (np.ndarray): the validation labels.
        X_test (np.ndarray): the testing features.
        y_test (np.ndarray): the testing labels.
    """
    # training the random forest model using the default parameters
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # saving the trained model
    with open(f"{os.path.join('.', 'files', 'random_forest.pkl')}", mode="wb") as f:
        pickle.dump(rf, f)

    # logging the artifacts
    client.log_artifact(run, "files/encoder.pkl")
    client.log_artifact(run, "files/random_forest.pkl")
    client.log_artifact(run, "files/preprocessed_test.csv")
    client.log_artifact(run, "files/preprocessed_train.csv")

    # calculating the metrics for the validation set
    valid_f1, valid_recall, valid_precision, valid_roc_auc = calculate_metrics(
        model=rf, y_true=y_valid, X=X_valid
    )

    # logging the metrics
    client.log_metric(run, "validation_f1_score", valid_f1)
    client.log_metric(run, "validation_recall", valid_recall)
    client.log_metric(run, "validation_precision", valid_precision)
    client.log_metric(run, "validation_roc_auc_score", valid_roc_auc)

    # calculating the metrics for the test set
    test_f1, test_recall, test_precision, test_roc_auc = calculate_metrics(
        model=rf, y_true=y_test, X=X_test
    )

    # logging the metrics
    client.log_metric(run, "test_f1_score", test_f1)
    client.log_metric(run, "test_recall", test_recall)
    client.log_metric(run, "test_precision", test_precision)
    client.log_metric(run, "test_roc_auc_score", test_roc_auc)


def models_tracking(
    client: MlflowClient,
    experiment_id: str,
) -> None:
    """
    Function responsible for tracking the models training step.

    Args:
        client (MlflowClient): the MlflowClient object.
        experiment_id (str): the experiment id.
    """
    # loading training set
    training_df = pd.read_csv("files/preprocessed_train.csv", sep=",")

    training_df, validation_df = train_test_split(
        training_df, test_size=0.2, shuffle=True, random_state=42
    )

    # loading testing set
    testing_df = pd.read_csv("files/preprocessed_test.csv", sep=",")

    # getting the mean embedding vector of each sample
    train_X, train_y = get_vector_mean(training_df)
    valid_X, valid_y = get_vector_mean(validation_df)
    test_X, test_y = get_vector_mean(testing_df)

    # creating the run that will be used to track the random forest model
    rf_run = client.create_run(experiment_id=experiment_id, run_name="random_forest")

    # training the random forest model and tracking its metrics and artifacts
    train_random_forest(
        client=client,
        run=rf_run.info.run_id,
        X_train=train_X,
        y_train=train_y,
        X_valid=valid_X,
        y_valid=valid_y,
        X_test=test_X,
        y_test=test_y,
    )

    # finishing the run
    client.set_terminated(run_id=rf_run.info.run_id)

    # creating the run that will be used to track the xgboost model
    xgb_run = client.create_run(experiment_id=experiment_id, run_name="xgboost")

    # training the xgboost model and tracking its metrics and artifacts
    train_xgboost(
        client=client,
        run=xgb_run.info.run_id,
        X_train=train_X,
        y_train=train_y,
        X_valid=valid_X,
        y_valid=valid_y,
        X_test=test_X,
        y_test=test_y,
    )

    # finishing the run
    client.set_terminated(run_id=xgb_run.info.run_id)

    # creating the run that will be used to track the bert model
    bert_run = client.create_run(experiment_id=experiment_id, run_name="bert")

    # training the bert model and tracking its metrics and artifacts
    train_bert_model(
        client=client,
        run=bert_run.info.run_id,
        train_df=training_df,
        valid_df=validation_df,
        test_df=testing_df,
    )

    # finishing the run
    client.set_terminated(run_id=bert_run.info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", "-t", action="store_true")
    parser.add_argument("--register", "-r", action="store_true")
    parser.add_argument("--run_id", type=str, required=False)
    parser.add_argument("--run_name", type=str, required=False)
    parser.add_argument("--name", type=str, required=False)
    parser.add_argument("--tags", type=str, required=False)
    parser.add_argument("--description", type=str, required=False)
    parser.add_argument("--version", type=str, required=False)
    args = parser.parse_args()

    # initializing the mlflow client
    client = MlflowClient()

    # preprocessing the data
    data_pipeline()

    try:
        experiment_id = client.create_experiment(
            name="first-mlflow-client-experiment", tags={"version": "v1"}
        )
    except mlflow.exceptions.MlflowException:
        experiment_id = client.get_experiment_by_name(
            "first-mlflow-client-experiment"
        ).experiment_id

    if args.tracking:
        models_tracking(client=client, experiment_id=experiment_id)
    elif args.register:
        assert (
            (not args.run_id is None)
            and (not args.run_name is None)
            and (not args.name is None)
            and (not args.tags is None)
            and (not args.description is None)
            and (not args.version is None)
        )

        # creating a new registry
        client.create_registered_model(name=args.name)

        # creating a new model version that will be saved into the model registry
        # that was created above
        result = client.create_model_version(
            source=f"runs:/{args.run_id}/{args.run_name}",
            name=args.name,
            tags=eval(args.tags),
            run_id=args.run_id,
            description=args.description,
        )

        print(result)
