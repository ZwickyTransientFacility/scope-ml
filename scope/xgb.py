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

    def predict(self, X, **kwargs):
        d = xgb.DMatrix(X)
        y_pred = self.model.predict(d)
        self.meta['y_pred'] = y_pred

        return y_pred

    def evaluate(self, X_test, y_test, **kwargs):
        dtest = xgb.DMatrix(X_test, label=y_test)

        y_pred = np.around(self.predict(X_test))

        # Generate confusion matrix
        self.meta['cm'] = confusion_matrix(y_test, y_pred)
        self.meta['y_test'] = y_test

        return self.model.eval(dtest, 'dtest', **kwargs)

    def load(self, path_model, **kwargs):
        self.model.load_model(path_model, **kwargs)

    def save(
        self,
        tag: str,
        output_path: str = "./",
        output_format: str = "json",
        plot: bool = False,
        **kwargs,
    ):
        if output_format not in ["json"]:
            raise ValueError("unknown output format")

        path = pathlib.Path(output_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        output_name = self.name if not tag else tag
        if not output_name.endswith('.json'):
            output_name += '.json'
        self.model.save_model(path / output_name)

        # Save diagnostic plots
        if plot:
            path = path / f"{tag}_plots"
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

            if self.meta['cm'] is not None:
                cname = tag.split('.')[0]
                make_confusion_matrix(
                    self.meta['cm'],
                    figsize=(8, 6),
                    cbar=False,
                    percent=False,
                    categories=['not ' + cname, cname],
                )
                sns.set_context('talk')
                plt.title(cname)
                plt.savefig(path / cmpdf, bbox_inches='tight')
                plt.close()

            y_test = self.meta.get('y_test', None)
            y_pred = self.meta.get('y_pred', None)

            if (y_test is not None) & (y_pred is not None):

                fpr, tpr, _ = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(y_test, y_pred)

                plot_roc(fpr, tpr, roc_auc)
                plt.savefig(path / rocpdf, bbox_inches='tight')
                plt.close()

                plot_pr(recall, precision)
                plt.savefig(path / recallpdf, bbox_inches='tight')
                plt.close()
