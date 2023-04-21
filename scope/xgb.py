import pathlib
from .models import AbstractClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
from scope.utils import make_confusion_matrix, plot_roc, plot_pr
import seaborn as sns
import numpy as np


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
        **kwargs,
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

        self.model = xgb.Booster(params=params)

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Evaluate on train and val sets
        evals = [(dtrain, 'dtrain'), (dval, 'dval')]

        self.meta['evals'] = evals

        self.model = xgb.train(
            self.meta['params'],
            dtrain,
            num_boost_round=self.meta['num_boost_round'],
            evals=self.meta['evals'],
            early_stopping_rounds=self.meta['early_stopping_rounds'],
            **kwargs,
        )

    def predict(self, X, name=None, **kwargs):
        d = xgb.DMatrix(X)
        y_pred = self.model.predict(d)

        if name is not None:
            self.meta[f'y_pred{name}'] = y_pred
        else:
            self.meta['y_pred'] = y_pred

        return y_pred

    def evaluate(self, X_eval, y_eval, name='test', **kwargs):
        d_eval = xgb.DMatrix(X_eval, label=y_eval)

        y_pred = np.around(self.predict(X_eval, name=f'_{name}'))

        self.meta[f'X_{name}'] = X_eval
        self.meta[f'y_{name}'] = y_eval

        # Generate confusion matrix
        self.meta[f'cm_{name}'] = confusion_matrix(y_eval, y_pred)

        return self.model.eval(d_eval, f'd{name}', **kwargs)

    def load(self, path_model, **kwargs):
        try:
            self.model.load_model(path_model, **kwargs)
        except Exception as e:
            print('Failure during model loading:')
            print(e)

    def save(
        self,
        tag: str,
        output_path: str = "./",
        output_format: str = "json",
        plot: bool = False,
        names: list = ['train', 'val', 'test'],
        **kwargs,
    ):
        if output_format not in ["json"]:
            raise ValueError("unknown output format")

        output_path = pathlib.Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        output_name = self.name if not tag else tag
        if not output_name.endswith('.json'):
            output_name += '.json'
        self.model.save_model(output_path / output_name)

        # Save diagnostic plots
        for name in names:
            if plot:
                path = output_path / f"{tag}_plots" / name
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                impvars = tag + '_impvars.pdf'
                cmpdf = tag + '_cm.pdf'
                recallpdf = tag + '_recall.pdf'
                rocpdf = tag + '_roc.pdf'

                max_num_features = kwargs.get('max_num_features', 8)

                _ = xgb.plot_importance(
                    self.model, max_num_features=max_num_features, grid=False
                )
                plt.title(tag + ' Feature Importance')
                plt.savefig(path / impvars, bbox_inches='tight')
                plt.close()

                if self.meta[f'cm_{name}'] is not None:
                    cname = tag.split('.')[0]
                    make_confusion_matrix(
                        self.meta[f'cm_{name}'],
                        figsize=(8, 6),
                        cbar=False,
                        percent=False,
                        categories=['not ' + cname, cname],
                    )
                    sns.set_context('talk')
                    plt.title(cname)
                    plt.savefig(path / cmpdf, bbox_inches='tight')
                    plt.close()

                y_compare = self.meta.get(f'y_{name}', None)
                y_pred = self.meta.get(f'y_pred_{name}', None)

                if (y_compare is not None) & (y_pred is not None):

                    fpr, tpr, _ = roc_curve(y_compare, y_pred)
                    roc_auc = auc(fpr, tpr)
                    precision, recall, _ = precision_recall_curve(y_compare, y_pred)

                    plot_roc(fpr, tpr, roc_auc)
                    plt.savefig(path / rocpdf, bbox_inches='tight')
                    plt.close()

                    plot_pr(recall, precision)
                    plt.savefig(path / recallpdf, bbox_inches='tight')
                    plt.close()
