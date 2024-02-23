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
from .utils import make_confusion_matrix, plot_roc, plot_pr
import seaborn as sns
import numpy as np
import json


class XGB(AbstractClassifier):
    """Baseline model with a statically-defined architecture"""

    def setup(
        self,
        max_depth=6,
        min_child_weight=1,
        eta=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=10,
        num_boost_round=999,
        scale_pos_weight=1.0,
        **kwargs,
    ):
        #  removed for now:
        #  'gpu_id': 0,
        #  'tree_method': 'gpu_hist',

        params = {
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "eta": eta,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": objective,
            "eval_metric": eval_metric,
            "scale_pos_weight": scale_pos_weight,
        }

        self.meta["early_stopping_rounds"] = early_stopping_rounds
        self.meta["num_boost_round"] = num_boost_round
        self.meta["params"] = params

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        seed = kwargs.get("seed", 42)
        nfold = kwargs.get("nfold", 5)
        metrics = kwargs.get("metrics", ["auc"])

        max_depth_start = kwargs.get("max_depth_start", 3)
        max_depth_stop = kwargs.get("max_depth_stop", 8)
        max_depth_step = kwargs.get("max_depth_step", 2)

        min_child_weight_start = kwargs.get("min_child_weight_start", 1)
        min_child_weight_stop = kwargs.get("min_child_weight_stop", 6)
        min_child_weight_step = kwargs.get("min_child_weight_step", 2)

        eta_list = kwargs.get("eta_list", [0.3, 0.2, 0.1, 0.05])

        subsample_start = kwargs.get("subsample_start", 6)
        subsample_stop = kwargs.get("subsample_stop", 11)
        subsample_step = kwargs.get("subsample_step", 2)

        colsample_bytree_start = kwargs.get("colsample_bytree_start", 6)
        colsample_bytree_stop = kwargs.get("colsample_bytree_stop", 11)
        colsample_bytree_step = kwargs.get("colsample_bytree_step", 2)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Evaluate on train and val sets
        evals = [(dtrain, "dtrain"), (dval, "dval")]
        self.meta["evals"] = evals

        skip_cv = kwargs.get("skip_cv", True)
        if not skip_cv:
            print("Running cross-validated hyperparameter grid search...")
            # Grid search for max_depth and min_child_weight params
            gridsearch_params = [
                (max_depth, min_child_weight)
                for max_depth in range(max_depth_start, max_depth_stop, max_depth_step)
                for min_child_weight in range(
                    min_child_weight_start, min_child_weight_stop, min_child_weight_step
                )
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
                self.meta["params"]["max_depth"] = max_depth
                self.meta["params"]["min_child_weight"] = min_child_weight
                # Run CV
                cv_results = xgb.cv(
                    self.meta["params"],
                    dtrain,
                    num_boost_round=self.meta["num_boost_round"],
                    seed=seed,
                    nfold=nfold,
                    metrics=metrics,
                    early_stopping_rounds=self.meta["early_stopping_rounds"],
                )
                # Update best AUC
                mean_auc = cv_results["test-auc-mean"].max()
                boost_rounds = cv_results["test-auc-mean"].argmax()
                print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
                if mean_auc > max_auc:
                    max_auc = mean_auc
                    best_params = (max_depth, min_child_weight)
            print(
                "Best params: {}, {}, AUC: {}".format(
                    best_params[0], best_params[1], max_auc
                )
            )
            self.meta["params"]["max_depth"] = best_params[0]
            self.meta["params"]["min_child_weight"] = best_params[1]

            # Grid search for subsample and colsample_bytree params
            gridsearch_params = [
                (subsample, colsample_bytree)
                for subsample in [
                    i / 10.0
                    for i in range(subsample_start, subsample_stop, subsample_step)
                ]
                for colsample_bytree in [
                    i / 10.0
                    for i in range(
                        colsample_bytree_start,
                        colsample_bytree_stop,
                        colsample_bytree_step,
                    )
                ]
            ]
            best_params = None
            max_auc = 0.0

            # We start by the largest values and go down to the smallest
            for subsample, colsample_bytree in reversed(gridsearch_params):
                print(
                    "CV with subsample={}, colsample_bytree={}".format(
                        subsample, colsample_bytree
                    )
                )
                # Update our parameters
                self.meta["params"]["subsample"] = subsample
                self.meta["params"]["colsample_bytree"] = colsample_bytree
                # Run CV
                cv_results = xgb.cv(
                    self.meta["params"],
                    dtrain,
                    num_boost_round=self.meta["num_boost_round"],
                    seed=seed,
                    nfold=nfold,
                    metrics=metrics,
                    early_stopping_rounds=self.meta["early_stopping_rounds"],
                )
                # Update best AUC
                mean_auc = cv_results["test-auc-mean"].max()
                boost_rounds = cv_results["test-auc-mean"].argmax()
                print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
                if mean_auc > max_auc:
                    max_auc = mean_auc
                    best_params = (subsample, colsample_bytree)
            print(
                "Best params: {}, {}, AUC: {}".format(
                    best_params[0], best_params[1], max_auc
                )
            )
            self.meta["params"]["subsample"] = best_params[0]
            self.meta["params"]["colsample_bytree"] = best_params[1]

            best_params = None
            max_auc = 0.0

            for eta in eta_list:
                print("CV with eta={}".format(eta))

                # We update our parameters
                self.meta["params"]["eta"] = eta

                # Run and time CV
                cv_results = xgb.cv(
                    self.meta["params"],
                    dtrain,
                    num_boost_round=self.meta["num_boost_round"],
                    seed=seed,
                    nfold=nfold,
                    metrics=metrics,
                    early_stopping_rounds=self.meta["early_stopping_rounds"],
                )

                # Update best AUC
                mean_auc = cv_results["test-auc-mean"].max()
                boost_rounds = cv_results["test-auc-mean"].argmax()
                print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
                if mean_auc > max_auc:
                    max_auc = mean_auc
                    best_params = eta
            print("Best params: {}, AUC: {}".format(best_params, max_auc))
            self.meta["params"]["eta"] = best_params

            # One more CV round for max_depth, min_child_weight params
            max_depth1 = self.meta["params"]["max_depth"] - max_depth_step
            max_depth2 = self.meta["params"]["max_depth"] + max_depth_step
            if max_depth1 < max_depth_start:
                max_depth1 = max_depth_start
            if max_depth2 > max_depth_stop - 1:
                max_depth2 = max_depth_stop - 1

            min_child_wt1 = (
                self.meta["params"]["min_child_weight"] - min_child_weight_step
            )
            min_child_wt2 = (
                self.meta["params"]["min_child_weight"] + min_child_weight_step
            )
            if min_child_wt1 < min_child_weight_start:
                min_child_wt1 = min_child_weight_start
            if min_child_wt2 > min_child_weight_stop - 1:
                min_child_wt2 = min_child_weight_stop - 1

            print(
                f"Running final grid search between max_depth of {max_depth1} and {max_depth2}, min_child_weight of {min_child_wt1} and {min_child_wt2} in steps of 1."
            )

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
                self.meta["params"]["max_depth"] = max_depth
                self.meta["params"]["min_child_weight"] = min_child_weight
                # Run CV
                cv_results = xgb.cv(
                    self.meta["params"],
                    dtrain,
                    num_boost_round=self.meta["num_boost_round"],
                    seed=seed,
                    nfold=nfold,
                    metrics=metrics,
                    early_stopping_rounds=self.meta["early_stopping_rounds"],
                )
                # Update best AUC
                mean_auc = cv_results["test-auc-mean"].max()
                boost_rounds = cv_results["test-auc-mean"].argmax()
                print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
                if mean_auc > max_auc:
                    max_auc = mean_auc
                    best_params = (max_depth, min_child_weight)
            print(
                "Best params: {}, {}, AUC: {}".format(
                    best_params[0], best_params[1], max_auc
                )
            )
            self.meta["params"]["max_depth"] = best_params[0]
            self.meta["params"]["min_child_weight"] = best_params[1]

            # One more CV round for subsample, colsample_bytree params
            subsample1 = int(self.meta["params"]["subsample"] * 10) - subsample_step
            subsample2 = int(self.meta["params"]["subsample"] * 10) + subsample_step
            if subsample1 < subsample_start:
                subsample1 = subsample_start
            if subsample2 > subsample_stop - 1:
                subsample2 = subsample_stop - 1

            colsample_bytree1 = (
                int(self.meta["params"]["colsample_bytree"] * 10)
            ) - colsample_bytree_step
            colsample_bytree2 = (
                int(self.meta["params"]["colsample_bytree"] * 10)
            ) + colsample_bytree_step
            if colsample_bytree1 < colsample_bytree_start:
                colsample_bytree1 = colsample_bytree_start
            if colsample_bytree2 > colsample_bytree_stop - 1:
                colsample_bytree2 = colsample_bytree_stop - 1

            print(
                f"Running final grid search between subsample of {subsample1} and {subsample2}, colsample_bytree of {colsample_bytree1} and {colsample_bytree2} in steps of 1."
            )

            gridsearch_params = [
                (subsample, colsample_bytree)
                for subsample in [i / 10.0 for i in range(subsample1, subsample2, 1)]
                for colsample_bytree in [
                    i / 10.0 for i in range(colsample_bytree1, colsample_bytree2, 1)
                ]
            ]

            # Define initial best params and max AUC
            best_params = None
            max_auc = 0.0

            for subsample, colsample_bytree in gridsearch_params:
                print(
                    "CV with subsample={}, colsample_bytree={}".format(
                        subsample, colsample_bytree
                    )
                )
                # Update our parameters
                self.meta["params"]["subsample"] = subsample
                self.meta["params"]["colsample_bytree"] = colsample_bytree
                # Run CV
                cv_results = xgb.cv(
                    self.meta["params"],
                    dtrain,
                    num_boost_round=self.meta["num_boost_round"],
                    seed=seed,
                    nfold=nfold,
                    metrics=metrics,
                    early_stopping_rounds=self.meta["early_stopping_rounds"],
                )
                # Update best AUC
                mean_auc = cv_results["test-auc-mean"].max()
                boost_rounds = cv_results["test-auc-mean"].argmax()
                print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
                if mean_auc > max_auc:
                    max_auc = mean_auc
                    best_params = (subsample, colsample_bytree)
            print(
                "Best params: {}, {}, AUC: {}".format(
                    best_params[0], best_params[1], max_auc
                )
            )
            self.meta["params"]["subsample"] = best_params[0]
            self.meta["params"]["colsample_bytree"] = best_params[1]

            print("Grid search complete.")

        # Train using optimized hyperparameters
        self.model = xgb.train(
            self.meta["params"],
            dtrain,
            num_boost_round=self.meta["num_boost_round"],
            evals=self.meta["evals"],
            early_stopping_rounds=self.meta["early_stopping_rounds"],
        )

        # One more iteration of training (stop at best iteration)
        self.meta["num_boost_round"] = self.model.best_iteration + 1
        self.model = xgb.train(
            self.meta["params"],
            dtrain,
            num_boost_round=self.meta["num_boost_round"],
            evals=self.meta["evals"],
        )

    def predict(self, X, name=None, **kwargs):
        d = xgb.DMatrix(X)
        y_pred = self.model.predict(d)

        if name is not None:
            self.meta[f"y_pred{name}"] = y_pred
        else:
            self.meta["y_pred"] = y_pred

        return y_pred

    def evaluate(self, X_eval, y_eval, name="test", **kwargs):
        d_eval = xgb.DMatrix(X_eval, label=y_eval)

        y_pred = np.around(self.predict(X_eval, name=f"_{name}"))

        self.meta[f"y_{name}"] = y_eval

        # Generate confusion matrix
        self.meta[f"cm_{name}"] = confusion_matrix(y_eval, y_pred, normalize="all")

        return self.model.eval(d_eval, f"d{name}", **kwargs)

    def load(self, path_model, **kwargs):
        self.model = xgb.Booster()

        plpath = pathlib.Path(path_model)
        name = pathlib.Path(plpath.name)
        parent = plpath.parent
        filename = name.with_suffix("")
        cfg_filename = str(filename) + ".params"

        try:
            # Load .json model file
            self.model.load_model(path_model, **kwargs)
            # Load .params file
            with open(str(parent / cfg_filename), "r") as f:
                cfg = json.load(f)
            # Convert config dict to str
            cfg = json.dumps(cfg)
            # Load optimal hyperparameters
            self.model.load_config(cfg)
        except Exception as e:
            print("Failure during model loading:")
            print(e)

    def save(
        self,
        tag: str,
        output_path: str = "./",
        output_format: str = "json",
        plot: bool = False,
        names: list = ["train", "val", "test"],
        cm_include_count=False,
        cm_include_percent=True,
        annotate_scores=False,
        **kwargs,
    ):
        if output_format not in ["json"]:
            raise ValueError("unknown output format")

        output_path = pathlib.Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        output_name = self.name if not tag else tag
        if not output_name.endswith(".json"):
            output_name += ".json"
        config_name = pathlib.Path(output_name).with_suffix("")
        config_name = str(config_name) + ".params"
        self.model.save_model(output_path / output_name)
        cfg = self.model.save_config()
        with open(output_path / config_name, "w") as f:
            f.write(cfg)

        stats_dct = {}

        # Save diagnostic plots
        for name in names:
            if plot:
                path = output_path / f"{tag}_plots" / name
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                impvars = tag + "_impvars.pdf"
                impvars_json = tag + "_impvars.json"
                cmpdf = tag + "_cm.pdf"
                recallpdf = tag + "_recall.pdf"
                rocpdf = tag + "_roc.pdf"
                stats_json = tag + "_stats.json"

                max_num_features = kwargs.get("max_num_features", 8)

                self.meta["importance"] = self.model.get_score()
                with open(path / impvars_json, "w") as f:
                    json.dump(self.meta["importance"], f)

                _ = xgb.plot_importance(
                    self.model,
                    max_num_features=max_num_features,
                    grid=False,
                    show_values=False,
                )
                plt.title(tag.split(".")[0])
                plt.savefig(path / impvars, bbox_inches="tight")
                plt.close()

                if self.meta[f"cm_{name}"] is not None:
                    cname = tag.split(".")[0]
                    accuracy, precision, recall, f1_score = make_confusion_matrix(
                        self.meta[f"cm_{name}"],
                        figsize=(8, 6),
                        cbar=False,
                        count=cm_include_count,
                        percent=cm_include_percent,
                        categories=["not " + cname, cname],
                        annotate_scores=annotate_scores,
                    )
                    stats_dct["accuracy"] = accuracy
                    stats_dct["precision"] = precision
                    stats_dct["recall"] = recall
                    stats_dct["f1_score"] = f1_score
                    sns.set_context("talk")
                    plt.title(cname)
                    plt.savefig(path / cmpdf, bbox_inches="tight")
                    plt.close()

                y_compare = self.meta.get(f"y_{name}", None)
                y_pred = self.meta.get(f"y_pred_{name}", None)

                if (y_compare is not None) & (y_pred is not None):

                    fpr, tpr, _ = roc_curve(y_compare, y_pred)
                    roc_auc = auc(fpr, tpr)
                    precision, recall, _ = precision_recall_curve(y_compare, y_pred)
                    stats_dct["roc_auc"] = roc_auc

                    plot_roc(fpr, tpr, roc_auc)
                    plt.savefig(path / rocpdf, bbox_inches="tight")
                    plt.close()

                    plot_pr(recall, precision)
                    plt.savefig(path / recallpdf, bbox_inches="tight")
                    plt.close()

                with open(path / stats_json, "w") as f:
                    json.dump(stats_dct, f)
