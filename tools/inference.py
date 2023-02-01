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
import pyarrow.dataset as ds
import scope
from scope.utils import (
    read_hdf,
    read_parquet,
    write_hdf,
    forgiving_true,
    impute_features,
    get_feature_stats,
)
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
JUST = 50


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

# Load training set
trainingSetPath = config['training']['dataset']
if trainingSetPath.endswith('.parquet'):
    trainingSet = read_parquet(trainingSetPath)
elif trainingSetPath.endswith('.h5'):
    trainingSet = read_hdf(trainingSetPath)
elif trainingSetPath.endswith('.csv'):
    trainingSet = pd.read_csv(trainingSetPath)
else:
    raise ValueError(
        'Training set must have one of .parquet, .h5 or .csv file formats.'
    )

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
        missing_dict[id.astype(str)] = []


def clean_data(
    features_df,
    feature_names,
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
                    missing_dict[id.astype(str)] = []
                missing_dict[id.astype(str)] += [feature]  # add feature to dict

    # impute missing values as specified in config
    features_df = impute_features(features_df, self_impute=True)

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
    write_csv: bool
        if True, write CSV file in addition to HDF5
    float_convert_types: 2-tuple of ints (16, 32 or 64)
        Existing and final float types for feature conversion
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
    write_csv = kwargs.get("write_csv", False)
    float_convert_types = kwargs.get("float_convert_types", (64, 32))
    feature_stats = kwargs.get("feature_stats", None)
    scale_features = kwargs.get("scale_features", "min_max")

    # default file location for source ids
    if whole_field:
        default_source_file = (
            BASE_DIR + "/../ids/field_" + str(field) + "/field_" + str(field) + ".h5"
        )
        default_features_file = BASE_DIR + "/../features/field_" + str(field)
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

    if verbose:
        # read source ids from hdf5 file
        ts = time.time()
        all_source_ids = np.array([])
        with h5py.File(source_ids_filename, "r") as f:
            ids = np.array(f[list(f.keys())[0]])
            all_source_ids = np.concatenate((all_source_ids, ids), axis=0)
        te = time.time()
        if tm:
            print(
                "read source_ids from .h5".ljust(JUST)
                + "\t --> \t"
                + str(round(te - ts, 4))
                + " s"
            )

        print("Number of ids:", len(all_source_ids))

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

    preds_collection = []
    ra_collection = np.array([])
    dec_collection = np.array([])
    period_collection = np.array([])
    source_id_count = 0

    ts = time.time()
    DS = ds.dataset(features_filename, format='parquet')
    generator = DS.to_batches()
    te = time.time()
    if tm:
        print(
            "read features from locally stored files".ljust(JUST)
            + "\t --> \t"
            + str(round(te - ts, 4))
            + " s"
        )

    batch_count = 0
    for batch in generator:
        batch_count += 1
        print(f'Batch {batch_count} of {len(DS.files)}...')
        features = batch.to_pandas()
        source_ids = features['_id'].values
        ra_collection = np.concatenate([ra_collection, features['ra'].values])
        dec_collection = np.concatenate([dec_collection, features['dec'].values])
        period_collection = np.concatenate(
            [period_collection, features['period'].values]
        )
        source_id_count += len(source_ids)

        try:
            features_metadata = json.loads(batch.schema.metadata[b'scope'])
        except KeyError:
            warnings.warn('Could not find existing metadata')
            features_metadata = {}

        if not xgbst:
            dmdt = np.expand_dims(
                np.array([d for d in features['dmdt'].apply(list).values]), axis=-1
            )

            if verbose:
                print("Features:\n", features)

            train_config = config["training"]["classes"][model_class]
            all_features = config["features"][train_config["features"]]
            feature_names = [
                key
                for key in all_features
                if forgiving_true(all_features[key]["include"])
            ]

            # Impute missing data and flag source ids containing missing values
            make_missing_dict(source_ids)
            features = clean_data(
                features,
                feature_names,
                field,
                ccd,
                quad,
                flag_ids,
                whole_field,
            )

            # Get feature stats using training set for scaling consistency
            if feature_stats is None:
                feature_stats = get_feature_stats(trainingSet, feature_names)
            elif feature_stats == 'config':
                feature_stats = config.get("feature_stats", None)

            if verbose:
                print("Computed feature stats:\n", feature_stats)

            # scale features
            ts = time.time()

            for feature in feature_names:
                stats = feature_stats.get(feature)
                if (stats is not None) and (stats["std"] != 0):
                    if scale_features == "median_std":
                        features[feature] = (
                            features[feature] - stats["median"]
                        ) / stats["std"]
                    elif scale_features == "min_max":
                        features[feature] = (features[feature] - stats["min"]) / (
                            stats["max"] - stats["min"]
                        )
                    else:
                        raise ValueError(
                            'Currently supported scaling methods are min_max and median_std.'
                        )
            te = time.time()
            if tm:
                print(
                    "min max scaling".ljust(JUST)
                    + "\t --> \t"
                    + str(round(te - ts, 4))
                    + " s"
                )

            ts = time.time()
            # Convert float64 to float32 to satisfy tensorflow requirements
            float_type_dict = {16: np.float16, 32: np.float32, 64: np.float64}
            float_init, float_final = float_convert_types[0], float_convert_types[1]

            features[
                features.select_dtypes(float_type_dict[float_init]).columns
            ] = features.select_dtypes(float_type_dict[float_init]).astype(
                float_type_dict[float_final]
            )
            # preds = model.predict([features[feature_names].values, dmdt])
            # Above: calling model.predict(features) in a loop leads to significant
            #        memory leak and eventual freezing
            #        (e.g. https://github.com/keras-team/keras/issues/13118,
            #        https://github.com/tensorflow/tensorflow/issues/44711)
            #
            # Replacing with model(features, training=False) produces the same output
            # but keeps memory usage under control.
            preds = model([features[feature_names].values, dmdt], training=False)
            preds = preds.numpy().flatten()
            features[model_class + '_dnn'] = preds
            te = time.time()
            if tm:
                print(
                    "dnn inference (model.predict())".ljust(JUST)
                    + "\t --> \t"
                    + str(round(te - ts, 4))
                    + " s"
                )
            features['Gaia_EDR3___id'] = (
                features['Gaia_EDR3___id'].fillna(0).astype(int)
            )
            features['AllWISE___id'] = features['AllWISE___id'].fillna(0).astype(int)
            features['PS1_DR1___id'] = features['PS1_DR1___id'].fillna(0).astype(int)
            preds_df = features[
                [
                    "_id",
                    "Gaia_EDR3___id",
                    "AllWISE___id",
                    "PS1_DR1___id",
                    model_class + '_dnn',
                ]
            ].round(2)
            preds_df.reset_index(inplace=True, drop=True)
        else:
            # xgboost inferencing
            train_config = path_model[-7]
            feature_names = config["inference"]["xgb"][train_config]
            features = clean_data(
                features,
                feature_names,
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

        preds_collection += [preds_df]

    preds_df = pd.concat(preds_collection, axis=0)
    preds_df.reset_index(drop=True, inplace=True)
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
        )
    else:
        default_outfile = out_dir + "field_" + str(field) + "/field_" + str(field)

    filename = kwargs.get("output", default_outfile)

    # Add metadata
    code_version = scope.__version__
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    preds_df.attrs['inference_scope_code_version'] = code_version
    preds_df.attrs['inference_dateTime_utc'] = start_dt
    preds_df.attrs.update(features_metadata)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.isfile(f'{filename}.h5'):
        # Merge existing and new preds
        existing_preds = read_hdf(f'{filename}.h5')
        existing_attrs = existing_preds.attrs
        existing_attrs.update(preds_df.attrs)
        preds_df = pd.merge(
            existing_preds,
            preds_df,
            on=['_id', 'Gaia_EDR3___id', 'AllWISE___id', 'PS1_DR1___id'],
            how='outer',
        )
        preds_df.attrs = existing_attrs
    else:
        # If new file, add ra/dec/period columns
        # Reorganize so inference columns are together, not interrupted by coords/period
        class_name = preds_df.columns[-1]
        inference_col = preds_df[class_name]
        preds_df.drop(columns=class_name, inplace=True)
        preds_df['ra'] = ra_collection
        preds_df['dec'] = dec_collection
        preds_df['period'] = period_collection
        preds_df[class_name] = inference_col

    write_hdf(preds_df, f'{filename}.h5')
    if write_csv:
        preds_df.to_csv(f'{filename}.csv', index=False)

    meta_filename = os.path.dirname(filename) + "/meta.json"
    if not os.path.exists(meta_filename):
        os.makedirs(os.path.dirname(meta_filename), exist_ok=True)
        dct = {}
        dct["field"] = field
        if xgbst:
            dct["xgb_models"] = [path_model]
        else:
            dct["dnn_models"] = [path_model]
        dct["total"] = source_id_count
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
