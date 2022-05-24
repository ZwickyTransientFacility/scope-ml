import fire
import numpy as np
import pandas as pd
import pathlib
from penquins import Kowalski
import warnings
from typing import List, Union
import yaml
import json
import os
import time
import h5py
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
JUST = 50


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

kowalski = Kowalski(
    token=config["kowalski"]["token"],
    protocol=config["kowalski"]["protocol"],
    host=config["kowalski"]["host"],
    port=config["kowalski"]["port"],
)


def gettime(f):
    '''
    Wrapper function that prints the time taken by a function f.
    '''

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(str(f.__name__).ljust(JUST) + "\t --> \t" + str(round(te - ts, 4)) + " s")
        return result

    return timed


missing_dict = {}


# @gettime
def make_missing_dict(source_ids):
    '''
    Make a dictionary for storing the missing features for all objects.
    '''
    for id in source_ids:
        missing_dict[id] = []


# @gettime
def clean_data(features_df, feature_names, feature_stats, ccd, quad, flag_ids=False):
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
    filename = "preds/ccd_" + str(ccd).zfill(2) + "_quad_" + str(quad) + "/flagged.json"
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
        stats = feature_stats.get(feature)  # get stats of feature from config.yaml

        # fill missing values with mean
        features_df[feature] = features_df[feature].fillna(stats['mean'])
    if flag_ids:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as outfile:
            try:
                json.dump(missing_dict, outfile)  # dump dictionary to a json file
            except Exception as e:
                print("error dumping flagged to json, message: ", e)
    return features_df


# @gettime
def get_features(
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR5",
    verbose: bool = False,
    **kwargs,
):
    ccd = kwargs.get("ccd", 1)
    quad = kwargs.get("quad", 1)
    features_path = kwargs.get("features_path", None)
    if features_path is not None:
        filename = features_path
    else:
        filename = (
            "preds/ccd_" + str(ccd).zfill(2) + "_quad_" + str(quad) + "/features.csv"
        )

    # Check if features are already stored
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        # df = pd.read_csv(dirname + filename)
        dmdt = np.expand_dims(np.array([d for d in df['dmdt'].values]), axis=-1)

    # query features if not already stored
    else:
        query_length = kwargs.get("query_length", 1000)

        if not hasattr(source_ids, "__iter__"):
            source_ids = (source_ids,)

        id = 0
        df_collection = []
        dmdt_collection = []
        while 1:
            limit = query_length
            skip = id * limit
            query = {
                "query_type": "find",
                "query": {
                    "catalog": features_catalog,
                    "filter": {"_id": {"$in": source_ids}},
                },
                "kwargs": {"limit": limit, "skip": skip},
            }
            response = kowalski.query(query=query)
            source_data = response.get("data")

            if source_data is None:
                print(response)
                raise ValueError(f"No data found for source ids {source_ids}")

            df_temp = pd.DataFrame.from_records(source_data)
            dmdt_temp = np.expand_dims(
                np.array([d for d in df_temp['dmdt'].values]), axis=-1
            )
            df_collection += [df_temp]
            dmdt_collection += [dmdt_temp]

            if ((id + 1) * query_length) > len(source_ids):
                break
            id += 1

        df = pd.concat(df_collection, axis=0)
        dmdt = np.vstack(dmdt_collection)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_pickle(filename)
        # df.to_csv(dirname + filename, index=False)

        if verbose:
            print(df)
            print("dmdt shape", dmdt.shape)

    return df, dmdt


# @gettime
def make_model(**kwargs):
    features_input = tf.keras.Input(
        shape=kwargs.get("features_input_shape", (40,)), name="features"
    )
    dmdt_input = tf.keras.Input(
        shape=kwargs.get("dmdt_input_shape", (26, 26, 1)), name="dmdt"
    )

    # dense branch to digest features
    x_dense = tf.keras.layers.Dropout(0.2)(features_input)
    x_dense = tf.keras.layers.Dense(256, activation='relu', name='dense_fc_1')(x_dense)
    x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
    x_dense = tf.keras.layers.Dense(32, activation='relu', name='dense_fc_2')(x_dense)

    # CNN branch to digest dmdt
    x_conv = tf.keras.layers.Dropout(0.2)(dmdt_input)
    x_conv = tf.keras.layers.SeparableConv2D(
        16, (3, 3), activation='relu', name='conv_conv_1'
    )(x_conv)
    # x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
    x_conv = tf.keras.layers.SeparableConv2D(
        16, (3, 3), activation='relu', name='conv_conv_2'
    )(x_conv)
    x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
    x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

    x_conv = tf.keras.layers.SeparableConv2D(
        32, (3, 3), activation='relu', name='conv_conv_3'
    )(x_conv)
    # x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
    x_conv = tf.keras.layers.SeparableConv2D(
        32, (3, 3), activation='relu', name='conv_conv_4'
    )(x_conv)
    x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
    # x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

    x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

    # concatenate
    x = tf.keras.layers.concatenate([x_dense, x_conv])
    x = tf.keras.layers.Dropout(0.4)(x)

    # one more dense layer?
    x = tf.keras.layers.Dense(16, activation='relu', name='fc_1')(x)

    # Logistic regression to output the final score
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='score')(x)

    m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

    return m


# @gettime
def run(
    path_model: Union[str, pathlib.Path],
    model_class: str,
    # source_ids: List[int],
    **kwargs,
):
    """Run SCoPe DNN model on a list of source ids

    $ python tools/inference.py \
      --path-model=models-prod/dr5-1/vnv-20210915_220725.h5 \
      --model-class=vnv \
      --source-ids=10487402158260,10487401023253

    $ python tools/inference.py \
      --path-model=models/vnv/20210915_220725/vnv.20210915_220725 \
      --model-class=vnv \
      --source-ids=10487402158260,10487401023253

    :param path_model:
    :param model_class:
    :param ccd:
    :param quad:
    :return:
    """

    # get arguments
    ccd = kwargs.get("ccd", 1)
    quad = kwargs.get("quad", 1)
    tm = kwargs.get("time", False)
    verbose = kwargs.get("verbose", False)
    flag_ids = kwargs.get("flag_ids", False)

    # default file location for source ids
    default_file = "output/data_ccd_" + str(ccd).zfill(2) + "_quad_" + str(quad) + ".h5"
    source_ids_filename = kwargs.get("source_ids_filename", default_file)

    filename = os.path.join(BASE_DIR, source_ids_filename)

    # read source ids from hdf5 file
    ts = time.time()
    source_ids = []
    with h5py.File(filename, "r") as f:
        for key in list(f.keys()):
            source_ids.append(list(map(int, f[key])))
    source_ids = [item for sublist in source_ids for item in sublist]
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
    features, dmdt = get_features(
        source_ids=source_ids,
        features_catalog="ZTF_source_features_DR5",
        verbose=verbose,
        ccd=ccd,
        quad=quad,
    )

    # scale features
    ts = time.time()
    train_config = config["training"]["classes"][model_class]
    feature_names = config["features"][train_config["features"]]
    feature_stats = config.get("feature_stats", None)
    scale_features = "min_max"

    for feature in feature_names:
        stats = feature_stats.get(feature)
        if (stats is not None) and (stats["std"] != 0):
            if scale_features == "median_std":
                features[feature] = (features[feature] - stats["median"]) / stats["std"]
            elif scale_features == "min_max":
                features[feature] = (features[feature] - stats["min"]) / (
                    stats["max"] - stats["min"]
                )
    te = time.time()
    if tm:
        print(
            "min max scaling".ljust(JUST) + "\t --> \t" + str(round(te - ts, 4)) + " s"
        )

    if verbose:
        print(features)

    # Load pre-trained model
    ts = time.time()
    if str(path_model).endswith(".h5"):
        model = tf.keras.models.load_model(path_model)
    else:
        model = make_model(features_input_shape=(len(feature_names),))
        if verbose:
            print(model.summary())
        model.load_weights(path_model).expect_partial()
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
    features = clean_data(features, feature_names, feature_stats, ccd, quad, flag_ids)

    ts = time.time()
    preds = model.predict([features[feature_names].values, dmdt])
    features[model_class] = preds
    te = time.time()
    if tm:
        print(
            "inference (model.predict())".ljust(JUST)
            + "\t --> \t"
            + str(round(te - ts, 4))
            + " s"
        )

    output_file = kwargs.get(
        "output",
        "preds/ccd_"
        + str(ccd).zfill(2)
        + "_quad_"
        + str(quad)
        + "/"
        + model_class
        + ".csv",
    )
    preds_df = features[["_id", model_class]]
    preds_df.reset_index(inplace=True, drop=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    preds_df.to_csv(output_file, index=False)
    # preds_df.to_pickle(output_file)

    if verbose:
        print("Predictions:\n", preds_df)


if __name__ == "__main__":
    fire.Fire(run)
