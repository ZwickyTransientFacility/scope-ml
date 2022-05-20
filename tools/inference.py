import fire
import numpy as np
import pandas as pd
import pathlib
from penquins import Kowalski
import tensorflow as tf
from typing import List, Union
import yaml
import warnings
import json
import os
import h5py

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('INFO')
BASE_DIR = os.path.dirname(__file__)


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

kowalski = Kowalski(
    token=config["kowalski"]["token"],
    protocol=config["kowalski"]["protocol"],
    host=config["kowalski"]["host"],
    port=config["kowalski"]["port"],
)

missing_dict = {}


def make_missing_dict(source_ids):
    for id in source_ids:
        missing_dict[id] = []


def clean_data(features_df, feature_names, feature_stats, flag_ids=None):
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
    flag_ids : str
        json file where flagged ids and features with missing values will be stored
    Returns
    -------
    Clean dataframe with no missing values.
    '''
    assert isinstance(features_df, pd.DataFrame), "df needs to be a pd.DataFrame"
    for feature in (
        features_df[feature_names]
        .columns[features_df[feature_names].isna().any()]
        .tolist()
    ):
        # print(feature, "stats mean", stats['mean']) # for debugging
        if flag_ids is not None:
            for id in features_df[features_df[feature].isnull()]['_id'].values:
                if id not in missing_dict.keys():
                    missing_dict[id.astype(int)] = []
                missing_dict[id.astype(int)] += [feature]
        stats = feature_stats.get(feature)
        features_df[feature] = features_df[feature].fillna(stats['mean'])
    if flag_ids is not None:
        with open(flag_ids, "w") as outfile:
            json.dump(missing_dict, outfile)
    # print(type(features_df), type(feature_names), type(feature_stats))
    return features_df


def get_features(
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR5",
    **kwargs,
):
    verbose = kwargs.get("verbose", False)
    query_length = kwargs.get("query_length", 1000)

    if not hasattr(source_ids, "__iter__"):
        source_ids = (source_ids,)

    id = 0
    df_collection = []
    dmdt_collection = []
    while 1:
        query = {
            "query_type": "find",
            "query": {
                "catalog": features_catalog,
                "filter": {
                    "_id": {
                        "$in": source_ids[
                            (id * query_length) : ((id + 1) * query_length)
                        ]
                    }
                },
            },
        }
        response = kowalski.query(query=query)
        source_data = response.get("data")

        if len(source_data) == 0:
            raise ValueError(f"No data found for source ids {source_ids}")

        df_temp = pd.DataFrame.from_records(source_data)
        df_collection += [df_temp]
        dmdt_temp = np.expand_dims(
            np.array([d for d in df_temp['dmdt'].values]), axis=-1
        )
        dmdt_collection += [dmdt_temp]

        if ((id + 1) * query_length) > len(source_ids):
            break
        id += 1

    df = pd.concat(df_collection, axis=0)
    dmdt = np.vstack(dmdt_collection)

    if verbose:
        print(df)
        print(dmdt.shape)

    return df, dmdt


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
    :param source_ids:
    :return:
    """
    source_ids_filename = kwargs.get(
        "source_ids_filename", "output/data_ccd_01_quad_1.h5"
    )
    filename = os.path.join(BASE_DIR, source_ids_filename)
    source_ids = []
    with h5py.File(filename, "r") as f:
        for key in list(f.keys()):
            source_ids.append(list(map(int, f[key])))
    # source_ids = source_ids[0]
    source_ids = [item for sublist in source_ids for item in sublist]

    verbose = kwargs.get("verbose", False)
    if verbose:
        print(len(source_ids))

    # source_ids = source_ids[:1000]
    # get raw features
    features, dmdt = get_features(source_ids)

    # scale features
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

    verbose = kwargs.get("verbose", False)
    if verbose:
        print(features)

    if str(path_model).endswith(".h5"):
        model = tf.keras.models.load_model(path_model)
    else:
        model = make_model(features_input_shape=(len(feature_names),))
        if verbose:
            print(model.summary())
        model.load_weights(path_model).expect_partial()

    make_missing_dict(source_ids)
    flag_ids = kwargs.get("flag_ids", None)
    features = clean_data(features, feature_names, feature_stats, flag_ids)

    # preds = model.predict([features[feature_names].values, dmdt])
    preds = model.predict(
        [
            np.asarray(features[feature_names].values).astype('float32'),
            np.asarray(dmdt).astype('float32'),
        ]
    )
    features[model_class] = preds
    if verbose:
        print(features[["_id", model_class]])

    output_file = kwargs.get("output", "preds.csv")
    features[["_id", model_class]].to_csv(output_file)


if __name__ == "__main__":
    fire.Fire(run)
