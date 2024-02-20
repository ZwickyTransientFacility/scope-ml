#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import warnings
import json
import os
import time
import pyarrow.dataset as ds
import scope
from scope.utils import (
    read_hdf,
    read_parquet,
    write_parquet,
    forgiving_true,
    impute_features,
    get_feature_stats,
    parse_load_config,
)
from scope.xgb import XGB
from datetime import datetime
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

BASE_DIR = pathlib.Path.cwd()
BASE_DIR_FEATS = BASE_DIR
BASE_DIR_PREDS = BASE_DIR
JUST = 50

config = parse_load_config()

path_to_features = config['feature_generation']['path_to_features']
path_to_preds = config['inference']['path_to_preds']

if path_to_features is not None:
    BASE_DIR_FEATS = pathlib.Path(path_to_features)
if path_to_preds is not None:
    BASE_DIR_PREDS = pathlib.Path(path_to_preds)

period_suffix_config = config['features']['info']['period_suffix']

# Load training set
trainingSetPath = BASE_DIR / config['training']['dataset']
if trainingSetPath.suffix == '.parquet':
    try:
        TRAINING_SET = read_parquet(trainingSetPath)
    except FileNotFoundError:
        warnings.warn('Training set file not found.')
        TRAINING_SET = []
elif trainingSetPath.suffix == '.h5':
    try:
        TRAINING_SET = read_hdf(trainingSetPath)
    except FileNotFoundError:
        warnings.warn('Training set file not found.')
        TRAINING_SET = []
elif trainingSetPath.suffix == '.csv':
    try:
        TRAINING_SET = pd.read_csv(trainingSetPath)
    except FileNotFoundError:
        warnings.warn('Training set file not found.')
        TRAINING_SET = []
else:
    raise ValueError(
        'Training set must have one of .parquet, .h5 or .csv file formats.'
    )

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
    algorithm='dnn',
    period_suffix=period_suffix_config,
):
    '''
    Impute missing values in features data
    Parameters
    ----------
    features_df : pd.DataFrame
        dataframe containing features of all sources (output of get_features)
    feature_names : List<str>
        features of interest for inference
    field : int
        ZTF field number
    ccd : int
        CCD number [1,16]
    quad : int
        CCD quad number [1,4]
    flag_ids : bool
        whether to store flagged ids and features with missing values
    whole_field : bool
        whether data to be cleaned comes from a single file for entire field
    algorithm : str
        'dnn' or 'xgb'
    period_suffix : str
        suffix to append to periodic feature names

    Returns
    -------
    Clean dataframe with no missing values.
    '''
    assert isinstance(features_df, pd.DataFrame), "df needs to be a pd.DataFrame"

    # file to store flagged ids and features with missing values
    if not whole_field:
        filename = (
            str(BASE_DIR_PREDS)
            + f"/preds_{algorithm}/field_"
            + str(field)
            + "/ccd_"
            + str(ccd).zfill(2)
            + "_quad_"
            + str(quad)
            + "_flagged.json"
        )
    else:
        filename = (
            str(BASE_DIR_PREDS)
            + f"/preds_{algorithm}/field_"
            + str(field)
            + f"/field_{field}_flagged.json"
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
    try:
        features_df = impute_features(
            features_df, self_impute=True, period_suffix=period_suffix
        )
    except KeyError:
        print(
            'KeyError during self-imputation - imputing features using training set specified in config.yaml'
        )
        print()
        features_df = impute_features(
            features_df, self_impute=False, period_suffix=period_suffix
        )

    if flag_ids:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as outfile:
            try:
                # dump dictionary to a json file
                json.dump(missing_dict, outfile)
            except Exception as e:
                print("error dumping flagged to json, message: ", e)
    return features_df


def run_inference(
    paths_models: list,
    model_class_names: list,
    field: str = '296',
    ccd: int = 1,
    quad: int = 1,
    whole_field: bool = False,
    flag_ids: bool = False,
    xgb_model: bool = False,
    verbose: bool = False,
    time_run: bool = False,
    write_csv: bool = False,
    float_convert_types: tuple = (64, 32),
    feature_stats: str = None,
    scale_features: str = 'min_max',
    trainingSet: str = 'use_config',
    feature_directory: str = 'features',
    feature_file_prefix: str = 'gen_features',
    period_suffix: str = period_suffix_config,
    no_write_metadata: bool = False,
    batch_size: int = 100000,
    **kwargs,
):
    """
    Run SCoPe models on a list of source ids

    Parameters
    ==========
    paths_model : list of str or pathlib.Path
        path(s) to model(s)
    model_class_name : list of str
        name(s) of model class(es)
    field : str
        field number
    ccd : int
        ccd number (with whole_field=False)
    quad : int
        quad number (with whole_field=False)
    whole_field : bool
        whether to run on whole field
    flag_ids : bool
        whether to flag ids having features with missing values
    xgb_model : bool
        evaluate using xgboost models
    verbose : bool
        whether to print progress
    time_run: bool
        print time taken by each step
    write_csv: bool
        if True, write CSV file in addition to parquet
    float_convert_types: 2-tuple of ints (16, 32 or 64)
        Existing and final float types for feature conversion
    feature_stats: str
        set to 'config' to read feature stats from config file
    scale_features: str
        method to use to scale features
    trainingSet: str
        usually set to 'use_config'. A DataFrame can also be passed in, but this is not recommended
    feature_directory: str
        name of directory containing features
    feature_file_prefix: str
        prefix of generated feature file names
    period_suffix: str
        suffix of column containing period to save with inference results
    no_write_metadata: bool
        if True, do not write metadata [useful for testing] (bool)
    batch_size: int
        batch size to use when reading feature files (int)

    Returns
    =======
    Stores the predictions at the following location:
        preds/field_<field>/field_<field>.csv

    USAGE:
    $ run-inference --paths-models models/dr5-1/agn-20210919_090902.h5 \
        --model-class agn --field 301 --ccd 1 --quad 1 --flag-ids

    $ run-inference --paths-models models/dr5-1/agn-20210919_090902.h5 \
        --model-class agn --field 301 --whole-field --flag-ids

    """

    if xgb_model:
        algorithm = 'xgb'
    else:
        algorithm = 'dnn'

    try:
        field = int(field)
        int_field = True
    except ValueError:
        int_field = False

    if not int_field:
        if 'specific_ids' in field:
            default_features_file = str(
                BASE_DIR_FEATS
                / f"{feature_directory}/specific_ids/{feature_file_prefix}_{field}.parquet"
            )
    else:
        # default file location for source ids
        if whole_field:
            default_features_file = (
                str(BASE_DIR_FEATS) + f"/{feature_directory}/field_" + str(field)
            )
        else:
            if feature_directory == 'features':
                default_features_file = (
                    str(BASE_DIR_FEATS)
                    + f"/{feature_directory}/field_"
                    + str(field)
                    + "/ccd_"
                    + str(ccd).zfill(2)
                    + "_quad_"
                    + str(quad)
                    + '.parquet'
                )
            else:
                default_features_file = (
                    str(BASE_DIR_FEATS)
                    + f"/{feature_directory}/field_"
                    + str(field)
                    + f"/{feature_file_prefix}_"
                    + "field_"
                    + str(field)
                    + "_ccd_"
                    + str(ccd)
                    + "_quad_"
                    + str(quad)
                    + '.parquet'
                )

    features_filename = kwargs.get("features_filename", default_features_file)

    out_dir = f"{str(BASE_DIR_PREDS)}/preds_{algorithm}/"

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

    source_id_count = 0
    ra_collection = np.array([])
    dec_collection = np.array([])
    period_collection = np.array([])
    field_collection = np.array([])
    ccd_collection = np.array([])
    quad_collection = np.array([])
    filter_collection = np.array([])
    gaia_edr3_id_collection = np.array([], dtype=np.int64)
    allwise_id_collection = np.array([], dtype=np.int64)
    ps1_dr1_id_collection = np.array([], dtype=np.int64)
    preds_collection = []
    filename = kwargs.get("output", default_outfile)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.isfile(f'{filename}.parquet'):
        warnings.warn(
            'Preds file for this field (/ccd/quad) already exists. Overwriting...'
        )

    ts = time.time()
    DS = ds.dataset(features_filename, format='parquet', exclude_invalid_files=True)
    generator = DS.to_batches(batch_size=batch_size)
    te = time.time()
    if time_run:
        print(
            "read features from locally stored files".ljust(JUST)
            + "\t --> \t"
            + str(round(te - ts, 4))
            + " s"
        )

    batch_count = 0
    max_batch_count = len(DS.files)

    if not ((period_suffix is None) | (period_suffix == 'None')):
        period_colname = f'period_{period_suffix}'
    else:
        period_colname = 'period'

    for batch in generator:
        batch_count += 1
        print(f'Batch {batch_count} of {max_batch_count}...')

        features = batch.to_pandas()
        source_ids = features['_id'].astype("Int64").values

        ra_collection = np.concatenate([ra_collection, features['ra'].values])
        dec_collection = np.concatenate([dec_collection, features['dec'].values])
        period_collection = np.concatenate(
            [period_collection, features[period_colname].values]
        )
        field_collection = np.concatenate(
            [field_collection, features['field'].astype("Int64").values]
        )
        ccd_collection = np.concatenate(
            [ccd_collection, features['ccd'].astype("Int64").values]
        )
        quad_collection = np.concatenate(
            [quad_collection, features['quad'].astype("Int64").values]
        )
        gaia_edr3_id_collection = np.concatenate(
            [
                gaia_edr3_id_collection,
                features['Gaia_EDR3___id'].fillna(0).astype("Int64"),
            ]
        )
        allwise_id_collection = np.concatenate(
            [allwise_id_collection, features['AllWISE___id'].fillna(0).astype("Int64")]
        )
        ps1_dr1_id_collection = np.concatenate(
            [ps1_dr1_id_collection, features['PS1_DR1___id'].fillna(0).astype("Int64")]
        )
        try:
            filter_collection = np.concatenate(
                [filter_collection, features['filter'].astype("Int64").values]
            )
            found_filters = True
        except KeyError:
            warnings.warn('Could not find filter column.')
            found_filters = False
        source_id_count += len(source_ids)

        try:
            features_metadata = json.loads(batch.schema.metadata[b'scope'])
        except KeyError:
            warnings.warn('Could not find existing metadata')
            features_metadata = {}

        # Fetch all features (under 'ontological' in config) to perform imputation once per batch file
        all_features = config["features"]["ontological"]
        feature_names = [
            key for key in all_features if forgiving_true(all_features[key]["include"])
        ]

        if not ((period_suffix is None) | (period_suffix == 'None')):
            periodic_bool = [all_features[x]['periodic'] for x in feature_names]
            for j, name in enumerate(feature_names):
                if periodic_bool[j]:
                    feature_names[j] = f'{name}_{period_suffix}'

        # Do not use dmdt as a feature for xgb algorithm
        if algorithm == 'xgb':
            if 'dmdt' in feature_names:
                feature_names.pop('dmdt')

        if verbose:
            print("Features:\n", features)

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
            algorithm=algorithm,
            period_suffix=period_suffix,
        )

        # Get feature stats using training set for scaling consistency
        if isinstance(trainingSet, str):
            if (trainingSet == 'use_config') & (len(TRAINING_SET) > 0):
                trainingSet = TRAINING_SET
            else:
                raise ValueError('Unable to find config-specified training set.')
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
                    features[feature] = (features[feature] - stats["median"]) / stats[
                        "std"
                    ]
                elif scale_features == "min_max":
                    features[feature] = (features[feature] - stats["min"]) / (
                        stats["max"] - stats["min"]
                    )
                elif scale_features is not None:
                    raise ValueError(
                        'Currently supported scaling methods are min_max, median_std, and None.'
                    )
        te = time.time()
        if time_run:
            print(
                "min max scaling".ljust(JUST)
                + "\t --> \t"
                + str(round(te - ts, 4))
                + " s"
            )

        for i, model_class in enumerate(model_class_names):
            print(f"{model_class}...")
            path_model = paths_models[i]

            # Load pre-trained model
            ts = time.time()
            if algorithm == 'dnn':
                model = tf.keras.models.load_model(path_model)
            elif algorithm == 'xgb':
                model = XGB(name=model_class)
                model.load(path_model)
            te = time.time()
            if time_run:
                print(
                    "load pre-trained model".ljust(JUST)
                    + "\t --> \t"
                    + str(round(te - ts, 4))
                    + " s"
                )

            # Redefine feature_names on a per-class basis (phenomenological or ontological)
            train_config = config["training"]["classes"][model_class]
            all_features = config["features"][train_config["features"]]

            feature_names = [
                key
                for key in all_features
                if forgiving_true(all_features[key]["include"])
            ]

            if not ((period_suffix is None) | (period_suffix == 'None')):
                periodic_bool = [all_features[x]['periodic'] for x in feature_names]
                for j, name in enumerate(feature_names):
                    if periodic_bool[j]:
                        feature_names[j] = f'{name}_{period_suffix}'

            if algorithm == 'dnn':
                dmdt = np.expand_dims(
                    np.array([d for d in features['dmdt'].apply(list).values]), axis=-1
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
                features[model_class + '_dnn'] = preds.round(2)

                te = time.time()
                if time_run:
                    print(
                        "dnn inference (model.predict())".ljust(JUST)
                        + "\t --> \t"
                        + str(round(te - ts, 4))
                        + " s"
                    )
            else:
                # xgboost inferencing
                ts = time.time()
                scores = model.predict(features[feature_names])
                features[model_class + '_xgb'] = scores.round(2)
                te = time.time()
                if time_run:
                    print(
                        "xgb inference (model.predict())".ljust(JUST)
                        + "\t --> \t"
                        + str(round(te - ts, 4))
                        + " s"
                    )
            if i == 0:
                preds_df = features[["_id", f"{model_class}_{algorithm}"]]
            else:
                preds_df[f"{model_class}_{algorithm}"] = features[
                    f"{model_class}_{algorithm}"
                ]

            if not no_write_metadata:
                meta_filename = os.path.dirname(filename) + "/meta.json"
                if batch_count == max_batch_count:
                    if not os.path.exists(meta_filename):
                        os.makedirs(os.path.dirname(meta_filename), exist_ok=True)
                        dct = {}
                        dct["field"] = field
                        dct[f"{algorithm}_models"] = [path_model]
                        dct["total"] = source_id_count
                    else:
                        with open(meta_filename, 'r') as f:
                            dct = json.load(f)
                            if f"{algorithm}_models" in dct.keys():
                                dct[f"{algorithm}_models"] += [path_model]

                    with open(meta_filename, "w") as f:
                        try:
                            json.dump(dct, f)  # dump dictionary to a json file
                        except Exception as e:
                            print("error dumping to json, message: ", e)

            preds_df.reset_index(inplace=True, drop=True)
            # End of one model_class

        preds_collection += [preds_df]
        # End of one batch file

    # Final preparation to save after all batch files done
    preds_df = pd.concat(preds_collection, axis=0)
    preds_df.reset_index(drop=True, inplace=True)

    # Add metadata
    code_version = scope.__version__
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    preds_df.attrs['inference_scope_code_version'] = code_version
    preds_df.attrs['inference_dateTime_utc'] = start_dt
    preds_df.attrs.update(features_metadata)

    # Add ids/ra/dec/period columns
    # Reorganize so inference columns are together, not interrupted by coords/period

    if 'fritz_name' in features.columns:
        preds_df['obj_id'] = features['fritz_name']

    preds_df['Gaia_EDR3___id'] = gaia_edr3_id_collection
    preds_df['AllWISE___id'] = allwise_id_collection
    preds_df['PS1_DR1___id'] = ps1_dr1_id_collection
    preds_df['ra'] = ra_collection
    preds_df['dec'] = dec_collection
    preds_df['period'] = period_collection
    preds_df['field'] = field_collection
    preds_df['ccd'] = ccd_collection
    preds_df['quad'] = quad_collection
    if found_filters:
        preds_df['filter'] = filter_collection

    for name in model_class_names:
        class_name = f"{name}_{algorithm}"
        inference_col = preds_df[class_name]
        preds_df.drop(columns=class_name, inplace=True)
        preds_df[class_name] = inference_col

    write_parquet(preds_df, f'{filename}.parquet')

    if write_csv:
        preds_df.to_csv(f'{filename}.csv', index=False)

    if verbose:
        print("Predictions:\n", preds_df)

    final_outfile = pathlib.Path(f'{filename}.parquet')
    return preds_df, final_outfile


def get_parser_minimal():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--paths-models",
        type=str,
        nargs='+',
        help="path(s) to model(s), space-separated",
    )
    parser.add_argument(
        "--model-class-names", type=str, nargs='+', help="name(s) of model class(es)"
    )
    parser.add_argument(
        "--whole-field", action='store_true', help="flag to run on whole field"
    )
    parser.add_argument(
        "--flag-ids",
        action='store_true',
        help="flag to flag ids having features with missing values",
    )
    parser.add_argument(
        "--xgb-model", action='store_true', help="flag to evaluate using XGBoost models"
    )
    parser.add_argument("--verbose", action='store_true', help="verbose flag")
    parser.add_argument(
        "--time-run",
        action='store_true',
        help="flag to time the inference run and print results",
    )
    parser.add_argument(
        "--write-csv",
        action='store_true',
        help="flag to write CSV file in addition to parquet",
    )
    parser.add_argument(
        "--float-convert-types",
        type=tuple,
        default=(64, 32),
        help="Existing and final float types for feature conversion",
    )
    parser.add_argument(
        "--feature-stats",
        type=str,
        default=None,
        help="set to 'config' to read feature stats from config file",
    )
    parser.add_argument(
        "--scale-features",
        type=str,
        default='min_max',
        help="method to use to scale features",
    )
    parser.add_argument(
        "--trainingSet",
        type=str,
        default='use_config',
        help="usually set to 'use_config'. A DataFrame can also be passed in, but this is not recommended.",
    )
    parser.add_argument(
        "--feature-directory",
        type=str,
        default='features',
        help="name of directory containing features",
    )
    parser.add_argument(
        "--feature-file-prefix",
        type=str,
        default='gen_features',
        help="prefix of feature filename",
    )
    parser.add_argument(
        "--period-suffix",
        type=str,
        default=period_suffix_config,
        help="suffix of column containing period to save with inference results",
    )
    parser.add_argument(
        "--no-write-metadata", action='store_true', help="flag to not write metadata"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="batch size to use when reading feature files",
    )
    return parser


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--paths-models",
        type=str,
        nargs='+',
        help="path(s) to model(s), space-separated",
    )
    parser.add_argument(
        "--model-class-names", type=str, nargs='+', help="name(s) of model class(es)"
    )
    parser.add_argument("--field", type=str, default='296', help="field number")
    parser.add_argument(
        "--ccd", type=int, default=1, help="ccd number (if whole_field is not set)"
    )
    parser.add_argument(
        "--quad", type=int, default=1, help="quad number (if whole_field is not set)"
    )
    parser.add_argument(
        "--whole-field", action='store_true', help="flag to run on whole field"
    )
    parser.add_argument(
        "--flag-ids",
        action='store_true',
        help="flag to flag ids having features with missing values",
    )
    parser.add_argument(
        "--xgb-model", action='store_true', help="flag to evaluate using XGBoost models"
    )
    parser.add_argument("--verbose", action='store_true', help="verbose flag")
    parser.add_argument(
        "--time-run",
        action='store_true',
        help="flag to time the inference run and print results",
    )
    parser.add_argument(
        "--write-csv",
        action='store_true',
        help="flag to write CSV file in addition to parquet",
    )
    parser.add_argument(
        "--float-convert-types",
        type=tuple,
        default=(64, 32),
        help="Existing and final float types for feature conversion",
    )
    parser.add_argument(
        "--feature-stats",
        type=str,
        default=None,
        help="set to 'config' to read feature stats from config file",
    )
    parser.add_argument(
        "--scale-features",
        type=str,
        default='min_max',
        help="method to use to scale features",
    )
    parser.add_argument(
        "--trainingSet",
        type=str,
        default='use_config',
        help="usually set to 'use_config'. A DataFrame can also be passed in, but this is not recommended.",
    )
    parser.add_argument(
        "--feature-directory",
        type=str,
        default='features',
        help="name of directory containing features",
    )
    parser.add_argument(
        "--feature-file-prefix",
        type=str,
        default='gen_features',
        help="prefix of feature filename",
    )
    parser.add_argument(
        "--period-suffix",
        type=str,
        default=period_suffix_config,
        help="suffix of column containing period to save with inference results",
    )
    parser.add_argument(
        "--no-write-metadata", action='store_true', help="flag to not write metadata"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="batch size to use when reading feature files",
    )
    return parser


def main():

    parser = get_parser()
    args, _ = parser.parse_known_args()

    run_inference(
        paths_models=args.paths_models,
        model_class_names=args.model_class_names,
        field=args.field,
        ccd=args.ccd,
        quad=args.quad,
        whole_field=args.whole_field,
        flag_ids=args.flag_ids,
        xgb_model=args.xgb_model,
        verbose=args.verbose,
        time_run=args.time_run,
        write_csv=args.write_csv,
        float_convert_types=args.float_convert_types,
        feature_stats=args.feature_stats,
        scale_features=args.scale_features,
        trainingSet=args.trainingSet,
        feature_directory=args.feature_directory,
        feature_file_prefix=args.feature_file_prefix,
        period_suffix=args.period_suffix,
        no_write_metadata=args.no_write_metadata,
        batch_size=args.batch_size,
    )
