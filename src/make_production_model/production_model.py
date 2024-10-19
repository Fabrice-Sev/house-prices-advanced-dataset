import mlflow
from mlflow.tracking import MlflowClient

# from conf import config

TRACKING_URI = "http://127.0.0.1:8080/"


def run_production_model(
    model_name_to_register: str = "Random_Forest_model",
    experiment_name: str = "House Pricing Exp",
    metric_for_selection: str = "rmse",
) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    id_metric_best_value = runs["metrics." + metric_for_selection].idxmax()
    best_run = runs.iloc[id_metric_best_value, :]
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    result = mlflow.register_model(
        f"runs:/{best_run.run_id}/model", f"{model_name_to_register}"
    )
    # client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    print(
        f"Model: {model_name_to_register} of run: {best_run['tags.mlflow.runName']} registered successfully!!"
    )
