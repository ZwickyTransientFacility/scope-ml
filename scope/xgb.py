import pathlib
from .models import AbstractClassifier
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
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
        seed = kwargs.get('seed', 42)
        nfold = kwargs.get('nfold', 5)
        metrics = kwargs.get('metrics', ['auc'])

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Evaluate on train and val sets
        evals = [(dtrain, 'dtrain'), (dval, 'dval')]
        self.meta['evals'] = evals

        # Grid search for max_depth and min_child_weight params
        gridsearch_params = [
            (max_depth, min_child_weight)
            for max_depth in range(3, 8, 2)
            for min_child_weight in range(1, 6, 2)
        ]
        # Define initial best params and max AUC
        best_params = None
        max_auc = 0.0

        for max_depth, min_child_weight in gridsearch_params:
            print(
                "CV with max_depth={}, min_child_weight={}".format(
                    max_depth, min_child_weight
                )
            )
            # Update our parameters
            self.meta['params']['max_depth'] = max_depth
            self.meta['params']['min_child_weight'] = min_child_weight
            # Run CV
            cv_results = xgb.cv(
                self.meta['params'],
                dtrain,
                num_boost_round=self.meta['num_boost_round'],
                seed=seed,
                nfold=nfold,
                metrics=metrics,
                early_stopping_rounds=self.meta['early_stopping_rounds'],
            )
            # Update best AUC
            mean_auc = cv_results['test-auc-mean'].max()
            boost_rounds = cv_results['test-auc-mean'].argmax()
            print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
            if mean_auc > max_auc:
                max_auc = mean_auc
                best_params = (max_depth, min_child_weight)
        print(
            "Best params: {}, {}, AUC: {}".format(
                best_params[0], best_params[1], max_auc
            )
        )
        self.meta['params']['max_depth'] = best_params[0]
        self.meta['params']['min_child_weight'] = best_params[1]

        # Grid search for subsample and colsample params
        gridsearch_params = [
            (subsample, colsample)
            for subsample in [i / 10.0 for i in range(6, 11, 2)]
            for colsample in [i / 10.0 for i in range(6, 11, 2)]
        ]
        best_params = None
        max_auc = 0.0

        # We start by the largest values and go down to the smallest
        for subsample, colsample in reversed(gridsearch_params):
            print("CV with subsample={}, colsample={}".format(subsample, colsample))
            # Update our parameters
            self.meta['params']['subsample'] = subsample
            self.meta['params']['colsample_bytree'] = colsample
            # Run CV
            cv_results = xgb.cv(
                self.meta['params'],
                dtrain,
                num_boost_round=self.meta['num_boost_round'],
                seed=seed,
                nfold=nfold,
                metrics=metrics,
                early_stopping_rounds=self.meta['early_stopping_rounds'],
            )
            # Update best AUC
            mean_auc = cv_results['test-auc-mean'].max()
            boost_rounds = cv_results['test-auc-mean'].argmax()
            print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
            if mean_auc > max_auc:
                max_auc = mean_auc
                best_params = (subsample, colsample)
        print(
            "Best params: {}, {}, AUC: {}".format(
                best_params[0], best_params[1], max_auc
            )
        )
        self.meta['params']['subsample'] = best_params[0]
        self.meta['params']['colsample_bytree'] = best_params[1]

        best_params = None
        max_auc = 0.0
        for eta in [0.3, 0.2, 0.1, 0.05]:
            print("CV with eta={}".format(eta))

            # We update our parameters
            self.meta['params']['eta'] = eta

            # Run and time CV
            cv_results = xgb.cv(
                self.meta['params'],
                dtrain,
                num_boost_round=self.meta['num_boost_round'],
                seed=seed,
                nfold=nfold,
                metrics=metrics,
                early_stopping_rounds=self.meta['early_stopping_rounds'],
            )

            # Update best AUC
            mean_auc = cv_results['test-auc-mean'].max()
            boost_rounds = cv_results['test-auc-mean'].argmax()
            print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
            if mean_auc > max_auc:
                max_auc = mean_auc
                best_params = eta
        print("Best params: {}, AUC: {}".format(best_params, max_auc))
        self.meta['params']['eta'] = best_params

        # max_depth 3, 5, 7
        # min_child_weight 1, 3, 5

        # Best 7, 1

        max_depth1 = self.meta['params']['max_depth'] - 1
        max_depth2 = self.meta['params']['max_depth'] + 2
        if max_depth1 < 1:
            max_depth1 = 1
        if max_depth2 > 9:
            max_depth2 = 9

        min_child_wt1 = self.meta['params']['min_child_weight'] - 1
        min_child_wt2 = self.meta['params']['min_child_weight'] + 2
        if min_child_wt1 < 1:
            min_child_wt1 = 1
        if min_child_wt2 > 9:
            min_child_wt2 = 9

        print(max_depth1, max_depth2, min_child_wt1, min_child_wt2)
        # 6 9 1 3

        # One more CV round for max_depth, min_child_weight params
        gridsearch_params = [
            (max_depth, min_child_weight)
            for max_depth in range(max_depth1, max_depth2, 1)
            for min_child_weight in range(min_child_wt1, min_child_wt2, 1)
        ]

        # Define initial best params and max AUC
        best_params = None
        max_auc = 0.0

        for max_depth, min_child_weight in gridsearch_params:
            print(
                "CV with max_depth={}, min_child_weight={}".format(
                    max_depth, min_child_weight
                )
            )
            # Update our parameters
            self.meta['params']['max_depth'] = max_depth
            self.meta['params']['min_child_weight'] = min_child_weight
            # Run CV
            cv_results = xgb.cv(
                self.meta['params'],
                dtrain,
                num_boost_round=self.meta['num_boost_round'],
                seed=seed,
                nfold=nfold,
                metrics=metrics,
                early_stopping_rounds=self.meta['early_stopping_rounds'],
            )
            # Update best AUC
            mean_auc = cv_results['test-auc-mean'].max()
            boost_rounds = cv_results['test-auc-mean'].argmax()
            print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
            if mean_auc > max_auc:
                max_auc = mean_auc
                best_params = (max_depth, min_child_weight)
        print(
            "Best params: {}, {}, AUC: {}".format(
                best_params[0], best_params[1], max_auc
            )
        )
        self.meta['params']['max_depth'] = best_params[0]
        self.meta['params']['min_child_weight'] = best_params[1]

        # Train using optimized hyperparameters
        self.model = xgb.train(
            self.meta['params'],
            dtrain,
            num_boost_round=self.meta['num_boost_round'],
            evals=self.meta['evals'],
            early_stopping_rounds=self.meta['early_stopping_rounds'],
            **kwargs,
        )

        # One more iteration of training (stop at best iteration)
        self.meta['num_boost_round'] = self.model.best_iteration + 1
        self.model = xgb.train(
            self.meta['params'],
            dtrain,
            num_boost_round=self.meta['num_boost_round'],
            evals=self.meta['evals'],
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
