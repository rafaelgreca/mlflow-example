# MLflow with Natural Language Processing

In this project, we trained a Random Forest, XGBoost, and a BERT model to predict movies' genres based on their descriptions using an IMDb dataset hosted on [Kaggle's website](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb?resource=download). Its main goal wasn't to develop the solution with the best accuracy among others developed by Kaggle's community but rather to understand the applicability of the MLflow framework and how to use it to track different models during the training and evaluation steps, while also learning about the model's versioning and registry. This project was deployed locally but can be extended to run on remote servers, such as AWS, GCP, Azure, and so on.


## Installation

To install this package, first clone the repository to the directory of your choice using the following command:

```bash
git clone https://github.com/rafaelgreca/mlflow-example.git
```

### Using Virtual Environment

Create a virtual environment (ideally using conda) and install the requirements with the following command:

```bash
conda create --name mlflow-env python=3.11.9
conda activate mflow-env
pip install -r requirements.txt
```

### Using Docker

Build the Docker image using the following command:

```bash
sudo docker build -f Dockerfile -t mlflow-docker-example .
```

### Using MLflow run

First, you need to build the Docker image using the previous command before continue. After that, you need to open two terminals and run both commands (one in each terminal):

```bash
mlflow server --host 127.0.0.1 --port 5000
```

and 

```python3
python3 run.py --experiment_name {EXPERIMENT_NAME}
```

## Examples

### Using MLflow's API

1. Run the tracking script using the command above. This will create a new MLflow experiment called `first-mlflow-experiment`, which will create an individual run for each model (one for the Random Forest, one for the XGBoost, and another for BERT) that uses the default hyperparameters for each model.

```python3
python3 tracking.py
```

2. After running the tracking step, you can visualize the results using the MLflow's UI with the following command.

```bash
mlflow ui --host 127.0.0.1
```

3. Register the best model (or whatever model you choose) using the next command. It's important to mention that you will need to pass a few parameters, which are named above:

* `--run_id`: the run ID of the model that you want to save.
* `--run_name`: the run name of the model that you want to save. 
* `--name`: the name you want to give to the model's registry.
* `--tags`: the tags you want to give to the model's registry.

```bash
python3 register.py --run_id {RUN_ID} \
                    --run_name {RUN_NAME} \
                    --name {REGISTRY_NAME} \
                    --tags {REGISTRY_TAGS}

```

4. After running the registry step, you can visualize the results using the MLflow's UI with the following command.

```bash
mlflow ui --host 127.0.0.1
```

**NOTE: Make sure to delete the `mlruns` folder before running the previous steps.**

### Using Mlflow's Client

1. Run the tracking script using the command above. This will create a new MLflow experiment called `first-mlflow-client-experiment` (works the same as with the MLflow's API script). It's important to mention that you will need to pass the `-t` parameters, which means that the tracking script will be used.

```python3
python3 mlflow_client.py -t
```

2. After running the tracking step, you can visualize the results using the MLflow's UI with the following command.

```bash
mlflow ui --host 127.0.0.1
```

3. Register the best model (or whatever model you choose) using the next command. It's important to mention that you will need to pass the `-r` parameter (which will run the script responsible for registring the desired model) with a few other parameters, which are named above:

* `--run_id`: the run ID of the model that you want to save.
* `--run_name`: the run name of the model that you want to save. 
* `--name`: the name you want to give to the model's registry.
* `--version`: the model's version.
* `--stage`: which stage the model is currently in.
* `--tags`: the tags you want to give to the model's registry.

```bash
python3 mlflow_client.py -r --run_id {RUN_ID} \
                    --run_name {RUN_NAME} \
                    --name {REGISTRY_NAME} \
                    --version {VERSION} \
                    --stage {STAGE} \
                    --tags {REGISTRY_TAGS}

```

4. After running the registry step, you can visualize the results using the MLflow's UI with the following command.

```bash
mlflow ui --host 127.0.0.1
```

**NOTE: Make sure to delete the `mlruns` folder before running the previous steps.**

## Feedback

If you have any feedback, please feel free to create an issue pointing out whatever you want or reach out to me at rgvieira97@gmail.com

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See LICENSE for more information.

## Authors

- [@rafaelgreca](https://www.github.com/rafaelgreca)