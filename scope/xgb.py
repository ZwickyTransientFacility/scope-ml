import pathlib
from .models import AbstractClassifier
import xgboost as xgb

# Removed unused imports for now


class XGB(AbstractClassifier):
    """Baseline model with a statically-defined architecture"""

    def setup(
        self,
        max_depth=6,
        min_child_weight=1,
        eta=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=10,
        num_boost_round=999,
        # **kwargs,
    ):
        #  removed for now:
        #  'gpu_id': 0,
        #  'tree_method': 'gpu_hist',

        params = {
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'eta': eta,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': objective,
            'eval_metric': eval_metric,
        }

        self.meta['early_stopping_rounds'] = early_stopping_rounds
        self.meta['num_boost_round'] = num_boost_round
        self.meta['params'] = params

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, **kwargs):

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Evaluate on train and val sets only
        evals = [(dtrain, 'dtrain'), (dval, 'dval'), (dtest, 'dtest')]

        self.meta['evals'] = evals

        self.model = xgb.train(
            self.meta['params'],
            dtrain,
            num_boost_round=self.meta['num_boost_round'],
            evals=self.meta['evals'],
            early_stopping_rounds=self.meta['early_stopping_rounds'],
            **kwargs,
        )

    def evaluate(self, test_dataset, **kwargs):
        # Relocate X_test and y_test to this method (instead of test_dataset)
        # Then use self.model.eval() to return the evaluation (takes dtest and name arguments like in training)
        pass
        # return self.model.evaluate(test_dataset, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def load(self, path_model, **kwargs):
        self.model.load_weights(path_model, **kwargs)

    def save(
        self,
        tag: str,
        output_path: str = "./",
        output_format: str = "h5",
    ):
        # Customize this based on how model is saved in notebook
        # .h5 format is preferable
        # Replace code below with XGB-specific saving

        if output_format not in ("h5",):
            raise ValueError("unknown output format")

        path = pathlib.Path(output_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        output_name = self.name if not tag else f"{self.name}.{tag}"
        if not output_name.endswith('.h5'):
            output_name += '.h5'
        self.model.save(path / output_name, save_format=output_format)
