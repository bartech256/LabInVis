class Evaluator:
    def __init__(self, model, cfg: Config):

    def evaluate(self, loader) -> Dict[str, float]:
        """Run model on test set, calculate MAE, RMSE, maybe R2."""