from src.utils import read_dataset, bert_preprocessing, get_vector_mean
from src.bert import BERT, train_bert, test_bert
from src.dataset import create_dataloader
from mlflow.models import infer_signature
from torch.nn.functional import one_hot
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict
import pandas as pd
import numpy as np
import mlflow
import pickle
import os
import torch


def data_pipeline() -> None:
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


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_example: Dict,
) -> str:
    """
    Function responsible for training the Random Forest model and then
    saving it results into the current running MLflow experiment.

    Args:
        X_train (np.ndarray): the training features array.
        y_train (np.ndarray): the training labels array.
        X_test (np.ndarray): the validation features array.
        y_test (np.ndarray): the validation labels array.
        input_example (Dict): the training dataframe in a Dict format that will be
            used to save the model's signatures.

    Returns:
        model_uri (str): the model uri.
    """
    # logging the random forest default parameters using autolog
    mlflow.sklearn.autolog(
        log_models=False,
        log_post_training_metrics=False,
        log_model_signatures=False,
        log_input_examples=False,
        log_datasets=False,
        silent=True,
    )

    # training the random forest model using the default parameters
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # logging only the metrics for the test set
    prediction_test = rf.predict(X_test)
    prediction_test_proba = rf.predict_proba(X_test)

    metrics = {
        "validation f1 score": f1_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "validation recall": recall_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "validation precision": precision_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "validation roc auc score": roc_auc_score(
            y_true=y_test,
            y_score=prediction_test_proba,
            average="weighted",
            multi_class="ovr",
        ),
    }

    mlflow.log_metrics(metrics)

    signature = infer_signature(model_input=X_test, model_output=prediction_test)

    # logging the artifacts
    model_uri = mlflow.sklearn.log_model(
        rf, "random_forest", signature=signature, input_example=input_example
    ).model_uri

    return model_uri


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_example: Dict,
) -> str:
    """
    Function responsible for training the XGBoost model and then
    saving it results into the current running MLflow experiment.

    Args:
        X_train (np.ndarray): the training features array.
        y_train (np.ndarray): the training labels array.
        X_test (np.ndarray): the validation features array.
        y_test (np.ndarray): the validation labels array.
        input_example (Dict): the training dataframe in a Dict format that will be
            used to save the model's signatures.

    Returns:
        model_uri (str): the model uri.
    """
    # logging the xgboost default parameters using autolog
    mlflow.xgboost.autolog(
        log_models=False,
        log_model_signatures=False,
        log_input_examples=False,
        log_datasets=False,
        silent=True,
    )

    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)

    # logging only the metrics for the test set
    prediction_test = xgb.predict(X_test)
    prediction_test_proba = xgb.predict_proba(X_test)

    metrics = {
        "validation f1 score": f1_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "validation recall": recall_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "validation precision": precision_score(
            y_true=y_test, y_pred=prediction_test, average="weighted", zero_division=0.0
        ),
        "validation roc auc score": roc_auc_score(
            y_true=y_test,
            y_score=prediction_test_proba,
            average="weighted",
            multi_class="ovr",
        ),
    }

    mlflow.log_metrics(metrics)

    signature = infer_signature(model_input=X_test, model_output=prediction_test)

    # logging the artifacts
    model_uri = mlflow.sklearn.log_model(
        xgb, "xgboost", signature=signature, input_example=input_example
    ).model_uri

    return model_uri


def train_bert_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, input_example: Dict
) -> str:
    """
    Function responsible for training the BERT model and then
    saving it results into the current running MLflow experiment.

    Args:
        train_df (np.ndarray): the training dataframe.
        test_df (np.ndarray): the validation dataframe.
        input_example (Dict): the training dataframe in a Dict format that will be
            used to save the model's signatures.

    Returns:
        model_uri (str): the model uri.
    """
    # defining global variables
    epochs = 4
    batch_size = 32
    max_len = 70
    lr = 2e-5

    # creating the Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # saving the tokenizer
    with open(f"{os.path.join(tokenizer, 'bert_tokenizer.pkl')}", mode="wb") as f:
        pickle.dump(tokenizer, f)

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
    mlflow.pytorch.autolog(
        log_models=False,
        log_datasets=False,
        silent=True,
    )

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

    signature = infer_signature(model_input=[test_ids.numpy(), test_attentions.numpy()])

    # logging the artifacts
    model_uri = mlflow.pytorch.log_model(
        model, "BERT", signature=signature, input_example=input_example
    ).model_uri

    return model_uri


if __name__ == "__main__":
    # creating an experiment
    experiment_id = mlflow.create_experiment(
        name="first-mlflow-experiment", tags={"version": "v1"}
    )

    # preprocessing the data
    data_pipeline()

    # loading training set
    training_df = pd.read_csv("files/preprocessed_train.csv", sep=",")
    example_input = training_df.to_dict()

    training_df, validation_df = train_test_split(
        training_df, test_size=0.2, shuffle=True, random_state=42
    )

    # loading testing set
    testing_df = pd.read_csv("files/preprocessed_test.csv", sep=",")

    # getting the mean embedding vector of each sample
    train_X, train_y = get_vector_mean(training_df)
    valid_X, valid_y = get_vector_mean(validation_df)
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

        # training the random forest model and saving it
        # into the mlflow experiment run
        rf_model_uri = train_random_forest(
            X_train=train_X,
            y_train=train_y,
            X_test=valid_X,
            y_test=valid_y,
            input_example=example_input,
        )

        mlflow.evaluate(
            model=rf_model_uri,
            data=test_X,
            targets=test_y,
            model_type="classifier",
            evaluators=["default"],
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

        # training the xgboost model and saving it
        # into the mlflow experiment run
        xgb_model_uri = train_xgboost(
            X_train=train_X,
            y_train=train_y,
            X_test=valid_X,
            y_test=valid_y,
            input_example=example_input,
        )

        mlflow.evaluate(
            model=xgb_model_uri,
            data=test_X,
            targets=test_y,
            model_type="classifier",
            evaluators=["default"],
        )

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

        # training the BERT model and saving it
        # into the mlflow experiment run
        bert_model_uri = train_bert_model(
            train_df=training_df, test_df=validation_df, input_example=example_input
        )

        # creating the Bert Tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        # preprocessing the testing texts to be exactly what Bert needs
        test_ids, test_attentions = bert_preprocessing(
            texts=testing_df["summary"].tolist(),
            max_len=70,
            tokenizer=tokenizer,
        )

        y_test = one_hot(torch.tensor(testing_df["genre"].tolist()))

        # creating the test dataloader
        test_dataloader = create_dataloader(
            input_ids=test_ids,
            attention_masks=test_attentions,
            labels=y_test,
            batch_size=32,
            num_workers=0,
            shuffle=False,
        )

        loaded_bert = mlflow.pytorch.load_model(bert_model_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_metrics, test_loss = test_bert(
            model=loaded_bert, device=device, dataloader=test_dataloader
        )

        loss = {"test loss": test_loss}

        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(loss)

        # mlflow.evaluate(
        #     model=bert_model_uri,
        #     data=testing_df["summary"].tolist(),
        #     targets=testing_df["genre"].tolist(),
        #     model_type="classifier",
        #     evaluators=["default"]
        # )
