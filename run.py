import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "127.0.0.1:5000"

if __name__ == "__main__":
    mlflow.projects.run(
        uri=".",
        entry_point="main",
        version="v1",
        parameters=None,
        experiment_name="mlflow-example"
    )