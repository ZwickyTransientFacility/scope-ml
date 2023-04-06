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
        # ADD A UNIQUE name = given by user
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


    def train(self, X_train, y_train, X_val, y_val, **kwargs):

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        #evals = [(dval, 'dval'), (dtest, 'dtest')]
        #Evaluate on train and val sets only:
        evals = [(dtrain, 'dtrain'), dval, 'dval')]

        self.meta['evals'] = evals

        self.model = xgb.train(
            self.params,
            dtrain,
            evals=self.meta['evals'],
            num_boost_round=self.meta['num_boost_round'],
            early_stopping_rounds=self.meta['early_stopping_rounds'],
            **kwargs,
        )
        
        #SAVE somehow:
        #We need to name each run a different thing:
        #self.name = xgb_run_{'input by user'}

    def evaluate(self, X_test, y_test,**kwargs):
        
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        test_dataset = [(dtest,'dtest')]
        self.meta['test_dataset'] = test_dataset

#New method: save stats separately so as not to overwrite self.model:
        stats = classifier.evaluate(
            self.params,
            dtest,
            self.meta['evals'],
            **kwargs,
        )
#I originally included predict and load because nn.py does,
               #but I don't know that xgb.py needs them for now
  #  def predict(self, x, **kwargs):
    #    return self.model.predict(x, **kwargs)
#
  #  def load(self, path_model, **kwargs):
     #   self.model.load_weights(path_model, **kwargs)

    def save(
        self,
        stats,
        tag: str,
        output_path: str = "./",
        output_format: str = "h5",
    ):
        # Customize this based on how model is saved in notebook
        # .h5 format is preferable

        #

        #Current method is a json file, added to throughout the notebook by the write function
        #Basically:
        best_fit = self.model
        with open(new_h5_format_file, "w") as outfile:
            json.dump(best_fit, outfile)
        # I really need to read how to create an h5 format, this is just the first notes for the evening!!

        #

        if output_format not in ("h5",):
            raise ValueError("unknown output format")

        path = pathlib.Path(output_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        

        output_name = self.name if not tag else f"{self.name}.{tag}"
        if not output_name.endswith('.h5'):
            output_name += '.h5'
        self.model.save(path / output_name, save_format=output_format)
