import mlflow
import argparse

mlflow.set_tracking_uri("http://127.0.0.1:5000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-n", type=str, required=True)
    args = parser.parse_args()

    mlflow.projects.run(
        uri=".",
        parameters=None,
        experiment_name=args.experiment_name
    )
