import mlflow
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-n", type=str, required=True)
    args = parser.parse_args()

    mlflow.projects.run(
        uri="https://github.com/rafaelgreca/mlflow-example",
        version="main",
        parameters=None,
        experiment_name=args.experiment_name,
    )
