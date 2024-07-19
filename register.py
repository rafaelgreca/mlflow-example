import mlflow
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--tags", type=str, required=True)
    args = parser.parse_args()

    result = mlflow.register_model(
        model_uri=f"runs:/{args.run_id}/{args.run_name}",
        name=args.name,
        tags=eval(args.tags),
        await_registration_for=150,
    )

    print(result)
