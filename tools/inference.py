#!/usr/bin/env python
import tensorflow as tf
import fire
import numpy as np
import pandas as pd
import pathlib
from penquins import Kowalski
import xgboost as xgb
import warnings
from typing import Union
import yaml
import json
import os
import time
import h5py
from scope.utils import read_parquet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
JUST = 50


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

# Use KowalskiInstances class here when approved
kowalski = Kowalski(
    token=config["kowalski"]["token"],
    protocol=config["kowalski"]["protocol"],
    host=config["kowalski"]["host"],
    port=config["kowalski"]["port"],
)


def clean_dataset_xgb(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


missing_dict = {}


def make_missing_dict(source_ids):
    '''
    Make a dictionary for storing the missing features for all objects.
    '''
    for id in source_ids:
        missing_dict[id] = []


def clean_data(
    features_df,
    feature_names,
    feature_stats,
    field,
    ccd,
    quad,
    flag_ids=False,
    whole_field=False,
):
    '''
    Impute missing values in features data
    Parameters
    ----------
    features_df : pd.DataFrame
        dataframe containing features of all sources (output of get_features)
    feature_names : List<str>
        features of interest for inference
    feature_stats : dict
        feature statistics from config.yaml
    flag_ids : bool
        whether to store flagged ids and features with missing values
    ccd : int
        CCD number [1,16]
    quad : int
        CCD quad number [1,4]
    Returns
    -------
    Clean dataframe with no missing values.
    '''
    assert isinstance(features_df, pd.DataFrame), "df needs to be a pd.DataFrame"

    # file to store flagged ids and features with missing values
    if not whole_field:
        filename = (
            BASE_DIR
            + "/../preds/field_"
            + str(field)
            + "/ccd_"
            + str(ccd).zfill(2)
            + "_quad_"
            + str(quad)
            + "_flagged.json"
        )
    else:
        filename = (
            BASE_DIR + "/../preds/field_" + str(field) + f"/field_{field}_flagged.json"
        )
    for feature in (
        features_df[feature_names]
        .columns[features_df[feature_names].isna().any()]
        .tolist()
    ):
        if flag_ids:
            for id in features_df[features_df[feature].isnull()]['_id'].values:
                if id not in missing_dict.keys():
                    missing_dict[id.astype(int)] = []
                missing_dict[id.astype(int)] += [feature]  # add feature to dict
        # get stats of feature from config.yaml
        stats = feature_stats.get(feature)

        # fill missing values with mean
        features_df[feature] = features_df[feature].fillna(stats['mean'])
    if flag_ids:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as outfile:
            try:
                # dump dictionary to a json file
                json.dump(missing_dict, outfile)
            except Exception as e:
                print("error dumping flagged to json, message: ", e)
    return features_df


# Previous model - delete when DNN is re-trained
# def make_model(**kwargs):
#     features_input = tf.keras.Input(
#         shape=kwargs.get("features_input_shape", (40,)), name="features"
#     )
#     dmdt_input = tf.keras.Input(
#         shape=kwargs.get("dmdt_input_shape", (26, 26, 1)), name="dmdt"
#     )
#
#     # dense branch to digest features
#     x_dense = tf.keras.layers.Dropout(0.2)(features_input)
#     x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_1')(x_dense)
#     x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
#     x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_2')(x_dense)
#
#     # CNN branch to digest dmdt
#     x_conv = tf.keras.layers.Dropout(0.2)(dmdt_input)
#     x_conv = tf.keras.layers.SeparableConv2D(
#         16, (3, 3), activation='relu', name='conv_conv_1'
#     )(x_conv)
#     # x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
#     x_conv = tf.keras.layers.SeparableConv2D(
#         16, (3, 3), activation='relu', name='conv_conv_2'
#     )(x_conv)
#     x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
#     x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
#
#     x_conv = tf.keras.layers.SeparableConv2D(
#         32, (3, 3), activation='relu', name='conv_conv_3'
#     )(x_conv)
#     # x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
#     x_conv = tf.keras.layers.SeparableConv2D(
#         32, (3, 3), activation='relu', name='conv_conv_4'
#     )(x_conv)
#     x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
#     # x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)
#
#     x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)
#
#     # concatenate
#     x = tf.keras.layers.concatenate([x_dense, x_conv])
#     x = tf.keras.layers.Dropout(0.4)(x)
#
#     # one more dense layer?
#     x = tf.keras.layers.Dense(16, activation='relu', name='fc_1')(x)
#
#     # Logistic regression to output the final score
#     x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)
#
#     m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)
#
#     return m


def run(
    path_model: Union[str, pathlib.Path],
    model_class: str,
    **kwargs,
):
    """
    Run SCoPe models on a list of source ids

    Parameters
    ==========
    path_model : str
        path to model
    model_class : str
        name of model class
    field : int
        field number
    whole_field : bool
        whether to run on whole field
    ccd : int
        ccd number (with whole_field=False)
    quad : int
        quad number (with whole_field=False)
    flag_ids : bool
        whether to flag ids having features with missing values
    xgb_model : bool
        evaluate using xgboost models
    verbose : bool
        whether to print progress
    time: bool
        print time taken by each step
    Returns
    =======
    Stores the predictions at the following location:
        preds/field_<field>/field_<field>.csv

    USAGE:
    $ python tools/inference.py --path-model=models/dr5-1/agn-20210919_090902.h5 \
        --model-class=agn --field=301 --ccd=1 --quad=1 --flag_ids

    $ python tools/inference.py --path-model=models/dr5-1/agn-20210919_090902.h5 \
        --model-class=agn --field=301 --whole-field --flag_ids

    """
    DEFAULT_FIELD = 302
    DEFAULT_CCD = 1
    DEFAULT_QUAD = 1
    DEFAULT_FLAG_IDS = False
    DEFAULT_XGBST = False

    # get arguments
    field = kwargs.get("field", DEFAULT_FIELD)
    ccd = kwargs.get("ccd", DEFAULT_CCD)
    quad = kwargs.get("quad", DEFAULT_QUAD)
    flag_ids = kwargs.get("flag_ids", DEFAULT_FLAG_IDS)
    xgbst = kwargs.get("xgb_model", DEFAULT_XGBST)
    tm = kwargs.get("time", False)
    verbose = kwargs.get("verbose", False)
    whole_field = kwargs.get("whole_field", False)

    # default file location for source ids
    if whole_field:
        default_source_file = (
            BASE_DIR + "/../ids/field_" + str(field) + "/field_" + str(field) + ".h5"
        )
        default_features_file = (
            BASE_DIR + "/../features/field_" + str(field) + "/field_" + str(field)
        )
    else:
        default_source_file = (
            BASE_DIR
            + "/../ids/field_"
            + str(field)
            + "/data_ccd_"
            + str(ccd).zfill(2)
            + "_quad_"
            + str(quad)
            + ".h5"
        )
        default_features_file = (
            BASE_DIR
            + "/../features/field_"
            + str(field)
            + "/ccd_"
            + str(ccd).zfill(2)
            + "_quad_"
            + str(quad)
        )
    source_ids_filename = kwargs.get("source_ids_filename", default_source_file)
    features_filename = kwargs.get("features_filename", default_features_file)

    source_ids_filename = os.path.join(BASE_DIR, source_ids_filename)
    features_filename = os.path.join(BASE_DIR, features_filename)

    # read source ids from hdf5 file
    ts = time.time()
    source_ids = np.array([])
    with h5py.File(source_ids_filename, "r") as f:
        ids = np.array(f[list(f.keys())[0]])
        source_ids = np.concatenate((source_ids, ids), axis=0)
    te = time.time()
    if tm:
        print(
            "read source_ids from .h5".ljust(JUST)
            + "\t --> \t"
            + str(round(te - ts, 4))
            + " s"
        )

    if verbose:
        print("Number of ids:", len(source_ids))

    # get raw features
    ts = time.time()
    features = read_parquet(features_filename + '.parquet')
    feature_stats = config.get("feature_stats", None)
    te = time.time()
    if tm:
        print(
            "read features from locally stored file".ljust(JUST)
            + "\t --> \t"
            + str(round(te - ts, 4))
            + " s"
        )

    if not xgbst:
        dmdt = np.expand_dims(
            np.array([d for d in features['dmdt'].apply(list).values]), axis=-1
        )

        # scale features
        ts = time.time()
        train_config = config["training"]["classes"][model_class]
        feature_names = config["features"][train_config["features"]]
        scale_features = "min_max"

        for feature in feature_names:
            stats = feature_stats.get(feature)
            if (stats is not None) and (stats["std"] != 0):
                if scale_features == "median_std":
                    features[feature] = (features[feature] - stats["median"]) / stats[
                        "std"
                    ]
                elif scale_features == "min_max":
                    features[feature] = (features[feature] - stats["min"]) / (
                        stats["max"] - stats["min"]
                    )
        te = time.time()
        if tm:
            print(
                "min max scaling".ljust(JUST)
                + "\t --> \t"
                + str(round(te - ts, 4))
                + " s"
            )

        if verbose:
            print(features)

        # Load pre-trained model
        ts = time.time()
        model = tf.keras.models.load_model(path_model)
        te = time.time()
        if tm:
            print(
                "load pre-trained model".ljust(JUST)
                + "\t --> \t"
                + str(round(te - ts, 4))
                + " s"
            )

        # Impute missing data and flag source ids containing missing values
        make_missing_dict(source_ids)
        features = clean_data(
            features,
            feature_names,
            feature_stats,
            field,
            ccd,
            quad,
            flag_ids,
            whole_field,
        )  # 1

        ts = time.time()
        preds = model.predict([features[feature_names].values, dmdt])
        features[model_class + '_dnn'] = preds
        te = time.time()
        if tm:
            print(
                "dnn inference (model.predict())".ljust(JUST)
                + "\t --> \t"
                + str(round(te - ts, 4))
                + " s"
            )
        preds_df = features[["_id", model_class + '_dnn']].round(2)
        preds_df.reset_index(inplace=True, drop=True)
    else:
        # xgboost inferencing
        train_config = path_model[-7]
        feature_names = config["inference"]["xgb"][train_config]
        features = clean_data(
            features,
            feature_names,
            feature_stats,
            field,
            ccd,
            quad,
            flag_ids,
            whole_field,
        )
        model = xgb.XGBRegressor()
        model.load_model(path_model)

        ts = time.time()
        scores = model.predict(features[feature_names])
        features[model_class + '_xgb' + train_config] = scores
        te = time.time()
        if tm:
            print(
                "dnn inference (model.predict())".ljust(JUST)
                + "\t --> \t"
                + str(round(te - ts, 4))
                + " s"
            )
        preds_df = features[["_id", model_class + '_xgb' + train_config]].round(2)
        preds_df.reset_index(inplace=True, drop=True)

    out_dir = os.path.join(os.path.dirname(__file__), f"{BASE_DIR}/../preds/")

    if not whole_field:
        default_outfile = (
            out_dir
            + "field_"
            + str(field)
            + "/ccd_"
            + str(ccd).zfill(2)
            + "_quad_"
            + str(quad)
            + ".csv"
        )
    else:
        default_outfile = (
            out_dir + "field_" + str(field) + "/field_" + str(field) + ".csv"
        )

    filename = kwargs.get("output", default_outfile)

    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        preds_df.to_csv(filename, index=False)
    else:
        preds_df.drop("_id", axis=1, inplace=True)
        df_temp = pd.read_csv(filename)
        df_temp = pd.concat([df_temp, preds_df], axis=1)
        df_temp.to_csv(filename, index=False)

    meta_filename = os.path.dirname(filename) + "/meta.json"
    if not os.path.exists(meta_filename):
        os.makedirs(os.path.dirname(meta_filename), exist_ok=True)
        dct = {}
        dct["field"] = field
        if xgbst:
            dct["xgb_models"] = [path_model]
        else:
            dct["dnn_models"] = [path_model]
        dct["total"] = len(source_ids)
    else:
        with open(meta_filename, 'r') as f:
            dct = json.load(f)
            if xgbst:
                if "xgb_models" in dct.keys():
                    dct["xgb_models"] += [path_model]
                else:
                    dct["xgb_models"] = [path_model]
            else:
                if "dnn_models" in dct.keys():
                    dct["dnn_models"] += [path_model]
                else:
                    dct["dnn_models"] = [path_model]
    with open(meta_filename, "w") as f:
        try:
            json.dump(dct, f)  # dump dictionary to a json file
        except Exception as e:
            print("error dumping to json, message: ", e)

    if verbose:
        print("Predictions:\n", preds_df)


if __name__ == "__main__":
    fire.Fire(run)
