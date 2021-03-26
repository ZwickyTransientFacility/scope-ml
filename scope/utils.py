__all__ = [
    "Dataset",
    "forgiving_true",
    "load_config",
    "log",
    "make_tdtax_taxonomy",
    "plot_gaia_hr",
    "plot_light_curve_data",
]

import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm.auto import tqdm
from typing import Mapping, Optional, Union
import yaml


def load_config(config_path: str):
    """
    Load config and secrets
    """
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    return config


def time_stamp():
    """

    :return: UTC time as a formatted string
    """
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")


def log(message: str):
    print(f"{time_stamp()}: {message}")


def forgiving_true(expression):
    return True if expression in ("t", "True", "true", "1", 1, True) else False


def make_tdtax_taxonomy(taxonomy: Mapping):
    """Recursively convert taxonomy definition from config["taxonomy"]
       into tdtax-parsable dictionary

    :param taxonomy: config["taxonomy"] section
    :return:
    """
    tdtax_taxonomy = dict()
    if taxonomy["class"] not in ("tds", "phenomenological", "ontological"):
        tdtax_taxonomy["name"] = f"{taxonomy['class']}: {taxonomy['name']}"
    else:
        tdtax_taxonomy["name"] = taxonomy["name"]
    if "subclasses" in taxonomy:
        tdtax_taxonomy["children"] = []
        for cls in taxonomy["subclasses"]:
            tdtax_taxonomy["children"].append(make_tdtax_taxonomy(cls))

    return tdtax_taxonomy


def plot_light_curve_data(
    light_curve_data: pd.DataFrame,
    period: Optional[float] = None,
    title: Optional[str] = None,
    save: Optional[str] = None,
):
    """Plot and save to file light curve data

    :param light_curve_data:
    :param period: float [days] if set, a phase-folded light curve will be displayed
    :param title: plot title
    :param save: path to save the plot
    :return:
    """
    plt.close("all")

    # Official start of ZTF MSIP survey, March 17, 2018
    jd_start = 2458194.5

    colors = {
        1: "#28a745",
        2: "#dc3545",
        3: "#00415a",
        "default": "#f3dc11",
    }

    mask_good_data = light_curve_data["catflags"] == 0
    df = light_curve_data.loc[mask_good_data]

    fig = plt.figure(figsize=(16, 9))
    if title is not None:
        fig.suptitle(title, fontsize=24)

    if period is None:
        ax1 = fig.add_subplot(111)
    else:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

    # plot different ZTF bands/filters
    for band in df["filter"].unique():
        mask_filter = df["filter"] == band
        ax1.errorbar(
            df.loc[mask_filter, "hjd"] - jd_start,
            df.loc[mask_filter, "mag"],
            df.loc[mask_filter, "magerr"],
            marker=".",
            color=colors[band],
            lw=0,
        )
        if period is not None:
            for n in [0, -1]:
                ax2.errorbar(
                    (df.loc[mask_filter, "hjd"] - jd_start) / period % 1 + n,
                    df.loc[mask_filter, "mag"],
                    df.loc[mask_filter, "magerr"],
                    marker=".",
                    color=colors[band],
                    lw=0,
                )
    # invert y axes since we are displaying magnitudes
    ax1.invert_yaxis()
    if period is not None:
        ax2.invert_yaxis()

    ax1.set_xlabel("Time")
    ax1.grid(lw=0.3)
    if period is not None:
        ax2.set_xlabel(f"phase [period={period:4.4g} days]")
        ax2.set_xlim(-1, 1)
        ax2.grid(lw=0.3)

    if save is not None:
        fig.tight_layout()
        plt.savefig(save)


def plot_gaia_hr(
    gaia_data: pd.DataFrame,
    path_gaia_hr_histogram: Union[str, pathlib.Path],
    title: Optional[str] = None,
    save: Optional[str] = None,
):
    """Plot the Gaia HR diagram with a sample of objects over-plotted

    source: https://vlas.dev/post/gaia-dr2-hrd/

    """
    # plot the H-R diagram for 1 M stars within 200 pc from the Sun
    plt.rc("text", usetex=True)

    # load background histogram
    histogram = np.loadtxt(path_gaia_hr_histogram)

    # make figure
    fig, ax = plt.subplots(figsize=(6, 6))
    if title is not None:
        fig.suptitle(title, fontsize=24)

    x_edges = np.arange(-0.681896, 5.04454978, 0.02848978)
    y_edges = np.arange(-2.90934, 16.5665952, 0.0968952)

    ax.pcolormesh(x_edges, y_edges, histogram.T, antialiased=False)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    ax.invert_yaxis()
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$M_G$")

    # plot sample data
    ax.errorbar(
        gaia_data["BP-RP"],
        gaia_data["M"],
        gaia_data["M"] - gaia_data["Ml"],
        marker=".",
        color="#e68a00",
        alpha=0.75,
        ls="",
        lw=0.5,
    )

    # display grid behind all other elements on the plot
    ax.set_axisbelow(True)
    ax.grid(lw=0.3)

    if save is not None:
        fig.tight_layout()
        plt.savefig(save)


""" Datasets """


class Dataset(object):
    def __init__(
        self,
        tag: str,
        path_dataset: str,
        features: tuple,
        verbose: bool = False,
        **kwargs,
    ):
        """Load csv file with the dataset containing both data and labels
        As of 20210317, it is produced by labels*.ipynb - this will change in a future PR

        :param tag:
        :param path_dataset:
        :param features:
        :param verbose:
        """
        self.verbose = verbose

        self.tag = tag
        self.features = features

        self.target = None

        if self.verbose:
            log(f"Loading {path_dataset}...")
        nrows = kwargs.get("nrows", None)
        self.df_ds = pd.read_csv(path_dataset, nrows=nrows)
        if self.verbose:
            log(self.df_ds[list(features)].describe())

        self.df_ds = self.df_ds.replace([np.inf, -np.inf, np.nan], 0.0)

        dmdt = []
        if self.verbose:
            print("Moving dmdt's to a dedicated numpy array...")
            iterator = tqdm(self.df_ds.itertuples(), total=len(self.df_ds))
        else:
            iterator = self.df_ds.itertuples()
        for i in iterator:
            data = np.array(json.loads(self.df_ds["dmdt"][i.Index]))
            if len(data.shape) == 0:
                dmdt.append(np.zeros((26, 26)))
            else:
                dmdt.append(data)

        self.dmdt = np.array(dmdt)
        self.dmdt = np.expand_dims(self.dmdt, axis=-1)

        # drop in df_ds:
        self.df_ds.drop(columns="dmdt")

    @staticmethod
    def threshold(a, t: float = 0.5):
        b = np.zeros_like(a)
        b[np.array(a) > t] = 1
        return b

    def make(
        self,
        target_label: str = "variable",
        threshold: float = 0.5,
        balance: Optional[float] = None,
        weight_per_class: bool = True,
        scale_features: str = "min_max",
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        feature_stats: Optional[dict] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 256,
        epochs: int = 300,
        **kwargs,
    ):
        """Make datasets for target_label

        :param target_label:
        :param threshold:
        :param balance:
        :param weight_per_class:
        :param scale_features: min_max | median_std
        :param test_size:
        :param val_size:
        :param random_state:
        :param feature_stats: feature_stats to use to standardize features.
                              if None, stats are computed from the data
        :param batch_size
        :param shuffle_buffer_size
        :param epochs
        :return:
        """

        # Note: Dataset.from_tensor_slices method requires the target variable to be of the int type.
        # TODO: see what to do about it when trying label smoothing in the future.

        target = np.asarray(
            list(map(int, self.threshold(self.df_ds[target_label].values, t=threshold)))
        )

        self.target = np.expand_dims(target, axis=1)

        neg, pos = np.bincount(target.flatten())
        total = neg + pos
        if self.verbose:
            log(
                f"Examples:\n  Total: {total}\n  Positive: {pos} ({100 * pos / total:.2f}% of total)\n"
            )

        w_pos = np.rint(self.df_ds[target_label].values) == 1
        index_pos = self.df_ds.loc[w_pos].index
        if target_label == "variable":
            # 'variable' is a special case: there is an explicit 'non-variable' label:
            w_neg = (
                np.asarray(
                    list(
                        map(
                            int,
                            self.threshold(
                                self.df_ds["non-variable"].values, t=threshold
                            ),
                        )
                    )
                )
                == 1
            )
        else:
            w_neg = ~w_pos
        index_neg = self.df_ds.loc[w_neg].index

        # balance positive and negative examples if there are more negative than positive?
        index_neg_dropped = None
        if balance:
            neg_sample_size = int(np.sum(w_pos) * balance)
            index_neg = (
                self.df_ds.loc[w_neg].sample(n=neg_sample_size, random_state=1).index
            )
            index_neg_dropped = self.df_ds.loc[
                list(set(self.df_ds.loc[w_neg].index) - set(index_neg))
            ].index

        ds_indexes = index_pos.to_list() + index_neg.to_list()

        # Train/validation/test split (we will use an 81% / 9% / 10% data split by default):

        train_indexes, test_indexes = train_test_split(
            ds_indexes, shuffle=True, test_size=test_size, random_state=random_state
        )
        train_indexes, val_indexes = train_test_split(
            train_indexes, shuffle=True, test_size=val_size, random_state=random_state
        )

        # Normalize features (dmdt's are already L2-normalized) (?using only the training samples?).
        # Obviously, the same norms will have to be applied at the testing and serving stages.

        # load/compute feature norms:
        if feature_stats is None:
            feature_stats = {
                feature: {
                    "min": np.min(self.df_ds.loc[ds_indexes, feature]),
                    "max": np.max(self.df_ds.loc[ds_indexes, feature]),
                    "median": np.median(self.df_ds.loc[ds_indexes, feature]),
                    "mean": np.mean(self.df_ds.loc[ds_indexes, feature]),
                    "std": np.std(self.df_ds.loc[ds_indexes, feature]),
                }
                for feature in self.features
            }
            if self.verbose:
                print("Computed feature stats:\n", feature_stats)

        # scale features
        for feature in self.features:
            stats = feature_stats.get("feature")
            if (stats is not None) and (stats["std"] != 0):
                if scale_features == "median_std":
                    self.df_ds[feature] = (
                        self.df_ds[feature] - stats["median"]
                    ) / stats["std"]
                elif scale_features == "min_max":
                    self.df_ds[feature] = (self.df_ds[feature] - stats["min"]) / (
                        stats["max"] - stats["min"]
                    )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.df_ds.loc[train_indexes, self.features].values,
                    "dmdt": self.dmdt[train_indexes],
                },
                target[train_indexes],
            )
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.df_ds.loc[val_indexes, self.features].values,
                    "dmdt": self.dmdt[val_indexes],
                },
                target[val_indexes],
            )
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.df_ds.loc[test_indexes, self.features].values,
                    "dmdt": self.dmdt[test_indexes],
                },
                target[test_indexes],
            )
        )
        dropped_negatives = (
            tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "features": self.df_ds.loc[
                            index_neg_dropped, self.features
                        ].values,
                        "dmdt": self.dmdt[index_neg_dropped],
                    },
                    target[index_neg_dropped],
                )
            )
            if balance
            else None
        )

        # Shuffle and batch the datasets:

        train_dataset = (
            train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat(epochs)
        )
        val_dataset = val_dataset.batch(batch_size).repeat(epochs)
        test_dataset = test_dataset.batch(batch_size)

        dropped_negatives = dropped_negatives.batch(batch_size) if balance else None

        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "dropped_negatives": dropped_negatives,
        }

        indexes = {
            "train": np.array(train_indexes),
            "val": np.array(val_indexes),
            "test": np.array(test_indexes),
            "dropped_negatives": np.array(index_neg_dropped.to_list())
            if index_neg_dropped is not None
            else None,
        }

        # How many steps per epoch?

        steps_per_epoch_train = len(train_indexes) // batch_size - 1
        steps_per_epoch_val = len(val_indexes) // batch_size - 1
        steps_per_epoch_test = len(test_indexes) // batch_size - 1

        steps_per_epoch = {
            "train": steps_per_epoch_train,
            "val": steps_per_epoch_val,
            "test": steps_per_epoch_test,
        }
        if self.verbose:
            print(f"Steps per epoch: {steps_per_epoch}")

        # Weight training data depending on the number of samples?
        # Very useful for imbalanced classification, especially in the cases with a small number of examples.

        if weight_per_class:
            # weight data class depending on number of examples?
            # num_training_examples_per_class = np.array([len(target) - np.sum(target), np.sum(target)])
            num_training_examples_per_class = np.array([len(index_neg), len(index_pos)])

            assert (
                0 not in num_training_examples_per_class
            ), "found class without any examples!"

            # fewer examples -- larger weight
            weights = (1 / num_training_examples_per_class) / np.linalg.norm(
                (1 / num_training_examples_per_class)
            )
            normalized_weight = weights / np.max(weights)

            class_weight = {i: w for i, w in enumerate(normalized_weight)}

        else:
            # working with binary classifiers only
            class_weight = {i: 1 for i in range(2)}

        return datasets, indexes, steps_per_epoch, class_weight
