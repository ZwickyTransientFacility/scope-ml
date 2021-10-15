import fire
import numpy as np
import pandas as pd
import pathlib
from penquins import Kowalski
import tensorflow as tf
from typing import List, Union
import yaml


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

kowalski = Kowalski(
    token=config["kowalski"]["token"],
    protocol=config["kowalski"]["protocol"],
    host=config["kowalski"]["host"],
    port=config["kowalski"]["port"],
)


def get_features(
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR5",
    **kwargs,
):
    verbose = kwargs.get("verbose", False)

    if not hasattr(source_ids, "__iter__"):
        source_ids = (source_ids,)

    query = {
        "query_type": "find",
        "query": {
            "catalog": features_catalog,
            "filter": {"_id": {"$in": source_ids}},
        },
    }
    response = kowalski.query(query=query)
    source_data = response.get("data")

    if len(source_data) == 0:
        raise ValueError(f"No data found for source ids {source_ids}")

    df = pd.DataFrame.from_records(source_data)
    dmdt = np.expand_dims(np.array([d for d in df['dmdt'].values]), axis=-1)
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
    source_ids: List[int],
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
    verbose = kwargs.get("verbose", False)
    if verbose:
        print(source_ids)

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

    preds = model.predict([features[feature_names].values, dmdt])
    features[model_class] = preds
    if verbose:
        print(features[["_id", model_class]])


if __name__ == "__main__":
    fire.Fire(run)
