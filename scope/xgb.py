import datetime
import os
import pathlib
from .models import AbstractClassifier

#!git clone https://github.com/DTrimarchi10/confusion_matrix.git
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, mean_absolute_error
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
#plot_confusion_matrix, is deprecated, so import ConfusionMatrixDisplay instead
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, average_precision_score
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import copy
import json

    

class XGB(AbstractClassifier):
    """Baseline model with a statically-defined architecture"""
    def __init__(
        self,
        **kwargs,
    )


#    def call(self, inputs, **kwargs):
#        features_input = inputs.get("features")
#        dtrain = xgb.DMatrix(X_train, label=y_train)
#        dtest = xgb.DMatrix(X_test, label=y_test)
#
#        return features_input, dtrain, dtest
#
    def summary(self, **kwargs):

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, "Test")],
            early_stopping_rounds=10
        )
        return model.summary()


    
    def setup(
        self,
        'max_depth':6,
        'min_child_weight': 1,
        'eta':.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        **kwargs,
    ):
      #  removed for now:
      #  'gpu_id': 0,
      #  'tree_method': 'gpu_hist',

    params = {'max_depth': max_depth,
    'min_child_weight': min_child_weight,
    'eta': eta,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'objective': objective,
  #  'gpu_id': gpu_id,
  #  'tree_method': tree_method,
    'eval_metric': eval_metric,
    }
    
    self.meta['params'] = params

    self.model.compile(
        params =self.meta["params"],
        train = self.dtrain,
        test = self.dtest,
        )

    def train(self,X_train, **kwargs):
        dtrain = xgb.DMatrix(X_train, label=y_train)

        return dtrain

    def test(self, X_test, **kwargs);
    
        dtest = xgb.DMatrix(X_test, label=y_test)

        return dtest

              
    def evaluate(self, test_dataset, **kwargs):
        return self.model.evaluate(test_dataset, **kwargs)

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

        if output_format not in ("h5",):
            raise ValueError("unknown output format")

        path = pathlib.Path(output_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        output_name = self.name if not tag else f"{self.name}.{tag}"
        if not output_name.endswith('.h5'):
            output_name += '.h5'
        self.model.save(path / output_name, save_format=output_format)

