import mlflow


class MLflowService:
    def __init__(self, tracking_uri: str = "file:./experiments") -> None:
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)

    def set_experiment(self, experiment_name: str) -> None:
        mlflow.set_experiment(experiment_name)
