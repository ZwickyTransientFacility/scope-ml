#!/usr/bin/env python
from contextlib import contextmanager
import datetime
from deepdiff import DeepDiff
import fire
import numpy as np
import os
import pandas as pd
import pathlib
from penquins import Kowalski
from pprint import pprint
import questionary
import subprocess
import sys
import tdtax
from tdtax import taxonomy  # noqa: F401
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Optional, Sequence, Union
import yaml

from scope.nn import DNN
from scope.utils import (
    Dataset,
    load_config,
    make_tdtax_taxonomy,
    plot_gaia_hr,
    plot_light_curve_data,
)


@contextmanager
def status(message):
    """
    Borrowed from https://github.com/cesium-ml/baselayer/

    :param message: message to print
    :return:
    """
    print(f"[·] {message}", end="")
    sys.stdout.flush()
    try:
        yield
    except Exception:
        print(f"\r[✗] {message}")
        raise
    else:
        print(f"\r[✓] {message}")


def check_configs(config_wildcards: Sequence = ("config.*yaml", )):
    """
    - Check if config files exist
    - Offer to use the config files that match the wildcards
    - For config.yaml, check its contents against the defaults to make sure nothing is missing/wrong

    :param config_wildcards:
    :return:
    """
    path = pathlib.Path(__file__).parent.absolute()

    for config_wildcard in config_wildcards:
        config = config_wildcard.replace("*", "")
        # use config defaults if configs do not exist?
        if not (path / config).exists():
            answer = questionary.select(
                f"{config} does not exist, do you want to use one of the following"
                " (not recommended without inspection)?",
                choices=[p.name for p in path.glob(config_wildcard)],
            ).ask()
            subprocess.run(["cp", f"{path / answer}", f"{path / config}"])

        # check contents of config.yaml WRT config.defaults.yaml
        if config == "config.yaml":
            with open(path / config.replace(".yaml", ".defaults.yaml")) as config_yaml:
                config_defaults = yaml.load(config_yaml, Loader=yaml.FullLoader)
            with open(path / config) as config_yaml:
                config_wildcard = yaml.load(config_yaml, Loader=yaml.FullLoader)
            deep_diff = DeepDiff(config_wildcard, config_defaults, ignore_order=True)
            difference = {
                k: v
                for k, v in deep_diff.items()
                if k in ("dictionary_item_added", "dictionary_item_removed")
            }
            if len(difference) > 0:
                print("config.yaml structure differs from config.defaults.yaml")
                pprint(difference)
                raise KeyError("Fix config.yaml before proceeding")


class Scope:

    def __init__(self):
        # check configuration
        with status("Checking configuration"):
            check_configs(config_wildcards=["config.*yaml"])

            self.config = load_config(
                pathlib.Path(__file__).parent.absolute() / "config.yaml"
            )

            # use token specified as env var (if exists)
            kowalski_token_env = os.environ.get("KOWALSKI_TOKEN")
            if kowalski_token_env is not None:
                self.config["kowalski"]["token"] = kowalski_token_env

        # try setting up K connection if token is available
        if self.config["kowalski"]["token"] is not None:
            with status("Setting up Kowalski connection"):
                self.kowalski = Kowalski(
                    token=self.config["kowalski"]["token"],
                    protocol=self.config["kowalski"]["protocol"],
                    host=self.config["kowalski"]["host"],
                    port=self.config["kowalski"]["port"],
                )
        else:
            self.kowalski = None
            # raise ConnectionError("Could not connect to Kowalski.")
            print("Kowalski not available")

    def _get_nearest_gaia(
            self,
            positions: Sequence[Sequence[float]],
            catalog: str = None,
            max_distance: Union[float, int] = 5.0,
            distance_units: str = "arcsec",
    ) -> pd.DataFrame:
        """Get nearest Gaia source for a set of given positions

        :param positions: R.A./Decl. [deg]
        :param catalog: Gaia catalog to query
        :param max_distance:
        :param distance_units: arcsec | arcmin | deg | rad
        :return:
        """
        if self.kowalski is None:
            raise ConnectionError("Kowalski connection not established.")
        if catalog is None:
            catalog = self.config["kowalski"]["collections"]["gaia"]
        query = {
            "query_type": "near",
            "query": {
                "max_distance": max_distance,
                "distance_units": distance_units,
                "radec": positions,
                "catalogs": {
                    catalog: {
                        "filter": {},
                        "projection": {
                            "parallax": 1,
                            "parallax_error": 1,
                            "pmra": 1,
                            "pmra_error": 1,
                            "pmdec": 1,
                            "pmdec_error": 1,
                            "phot_g_mean_mag": 1,
                            "phot_bp_mean_mag": 1,
                            "phot_rp_mean_mag": 1,
                            "ra": 1,
                            "dec": 1,
                        }
                    }
                }
            },
            "kwargs": {
                "limit": 1
            }
        }
        response = self.kowalski.query(query=query)
        gaia_nearest = [
            v[0] for k, v in response.get("data").get(catalog).items()
            if len(v) > 0
        ]
        df = pd.DataFrame.from_records(gaia_nearest)

        df["M"] = df["phot_g_mean_mag"] + 5 * np.log10(df["parallax"] * 0.001) + 5
        df["Ml"] = df["phot_g_mean_mag"] + 5 * np.log10((df["parallax"] + df["parallax_error"]) * 0.001) + 5
        df["BP-RP"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]

        return df

    def _get_light_curve_data(
        self,
        ra: float,
        dec: float,
        catalog: str = "ZTF_sources_20201201",
        cone_search_radius: Union[float, int] = 2,
        cone_search_unit: str = "arcsec",
        filter_flagged_data: bool = True,
    ) -> pd.DataFrame:
        """Get light curve data from Kowalski

        :param ra: R.A. in deg
        :param dec: Decl. in deg
        :param catalog: collection name on Kowalski
        :param cone_search_radius:
        :param cone_search_unit: arcsec | arcmin | deg | rad
        :param filter_flagged_data: remove flagged/bad data?
        :return: flattened light curve data as pd.DataFrame
        """
        if self.kowalski is None:
            raise ConnectionError("Kowalski connection not established.")
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "cone_search_radius": cone_search_radius,
                    "cone_search_unit": cone_search_unit,
                    "radec": {
                        "target": [ra, dec]
                    }
                },
                "catalogs": {
                    catalog: {
                        "filter": {},
                        "projection": {
                            "_id": 1,
                            "filter": 1,
                            "data.hjd": 1,
                            "data.fid": 1,
                            "data.mag": 1,
                            "data.magerr": 1,
                            "data.ra": 1,
                            "data.dec": 1,
                            "data.programid": 1,
                            "data.catflags": 1
                        }
                    }
                }
            }
        }
        response = self.kowalski.query(query=query)
        light_curves_raw = response.get("data").get(catalog).get("target")

        light_curves = []
        for light_curve in light_curves_raw:
            df = pd.DataFrame.from_records(light_curve["data"])
            # broadcast to all data points:
            df["_id"] = light_curve["_id"]
            df["filter"] = light_curve["filter"]
            light_curves.append(df)

        df = pd.concat(light_curves, ignore_index=True)

        if filter_flagged_data:
            mask_flagged_data = df["catflags"] != 0
            df = df.loc[~mask_flagged_data]

        return df

    @staticmethod
    def develop():
        """Install developer tools"""
        subprocess.run(["pre-commit", "install"])

    @classmethod
    def lint(cls):
        """Lint sources"""
        try:
            import pre_commit  # noqa: F401
        except ImportError:
            cls.develop()

        try:
            subprocess.run(["pre-commit", "run", "--all-files"], check=True)
        except subprocess.CalledProcessError:
            sys.exit(1)

    def doc(self):
        """Build docs"""

        # generate taxonomy.html
        with status("Generating taxonomy visualization"):
            path_static = pathlib.Path(__file__).parent.absolute() / "doc" / "_static"
            if not path_static.exists():
                path_static.mkdir(parents=True, exist_ok=True)
            tdtax.write_viz(
                make_tdtax_taxonomy(self.config["taxonomy"]),
                outname=path_static / "taxonomy.html"
            )

        # generate images for the Field Guide
        if (self.kowalski is None) or (not self.kowalski.ping()):
            print("Kowalski connection not established, cannot generate docs.")
            return

        # example light curves
        with status("Generating example light curves"):
            path_doc_data = pathlib.Path(__file__).parent.absolute() / "doc" / "data"

            for sample_object_name, sample_object in self.config["docs"]["field_guide"].items():
                sample_light_curves = self._get_light_curve_data(
                    ra=sample_object["coordinates"][0],
                    dec=sample_object["coordinates"][1],
                    catalog=self.config["kowalski"]["collections"]["sources"],
                )
                plot_light_curve_data(
                    light_curve_data=sample_light_curves,
                    period=sample_object["period"],
                    title=sample_object["title"],
                    save=path_doc_data / sample_object_name,
                )

        # example HR diagrams for all Golden sets
        with status("Generating HR diagrams for Golden sets"):
            path_gaia_hr_histogram = (
                pathlib.Path(__file__).parent.absolute() / "doc" / "data" / "gaia_hr_histogram.dat"
            )
            # stored as ra/decs in csv format under /data/golden
            golden_sets = pathlib.Path(__file__).parent.absolute() / "data" / "golden"
            for golden_set in golden_sets.glob("*.csv"):
                golden_set_name = golden_set.stem
                positions = pd.read_csv(golden_set).to_numpy().tolist()
                gaia_sources = self._get_nearest_gaia(positions=positions)

                plot_gaia_hr(
                    gaia_data=gaia_sources,
                    path_gaia_hr_histogram=path_gaia_hr_histogram,
                    save=path_doc_data / f"hr__{golden_set_name}",
                )

        # build docs
        subprocess.run(["make", "html"], cwd="doc", check=True)

    @staticmethod
    def fetch_models():
        """
        Fetch SCoPe models from GCP

        :return:
        """
        path_models = pathlib.Path(__file__).parent / "models"
        if not path_models.exists():
            path_models.mkdir(parents=True, exist_ok=True)

        command = [
            "gsutil",
            "-m",
            "cp",
            "-n",
            "-r",
            "gs://ztf-scope/models/*",
            str(path_models),
        ]
        p = subprocess.run(command, check=True)
        if p.returncode != 0:
            raise RuntimeError("Failed to fetch SCoPe models")

    @staticmethod
    def fetch_datasets():
        """
        Fetch SCoPe datasets from GCP

        :return:
        """
        path_datasets = pathlib.Path(__file__).parent / "data" / "training"
        if not path_datasets.exists():
            path_datasets.mkdir(parents=True, exist_ok=True)

        command = [
            "gsutil",
            "-m",
            "cp",
            "-n",
            "-r",
            "gs://ztf-scope/datasets/*",
            str(path_datasets),
        ]
        p = subprocess.run(command, check=True)
        if p.returncode != 0:
            raise RuntimeError("Failed to fetch SCoPe datasets")

    def train(
        self,
        tag: str,
        path_dataset: str,
        gpu: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """Train classifier

        :param tag:
        :param path_dataset:
        :param gpu: GPU id to use, zero-based. check tf.config.list_physical_devices('GPU') for available devices
        :param verbose:
        :param kwargs:
        :return:
        """

        train_config = self.config["training"]["classes"][tag]

        features = self.config["features"][train_config["features"]]

        ds = Dataset(
            tag=tag,
            path_dataset=path_dataset,
            features=features,
            verbose=verbose
        )

        label = train_config["label"]

        # values from kwargs override those defined in config. if latter is absent, use reasonable default
        threshold = kwargs.get("threshold", train_config.get("threshold", 0.5))
        balance = kwargs.get("balance", train_config.get("balance", None))
        weight_per_class = kwargs.get("weight_per_class", train_config.get("weight_per_class", False))

        test_size = kwargs.get("test_size", 0.1)
        val_size = kwargs.get("val_size", 0.1)
        random_state = kwargs.get("random_state", 42)
        norms = self.config.get("feature_norms", None)

        batch_size = kwargs.get("batch_size", 32)
        shuffle_buffer_size = kwargs.get("shuffle_buffer_size", 512)
        epochs = kwargs.get("epochs", 200)

        datasets, indexes, steps_per_epoch, class_weight = ds.make(
            target_label=label,
            threshold=threshold,
            balance=balance,
            weight_per_class=weight_per_class,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            norms=norms,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            epochs=epochs,
        )

        # set up and train model

        if gpu is not None:
            # specified a GPU to run on?
            gpus = tf.config.list_physical_devices("GPU")
            tf.config.experimental.set_visible_devices(gpus[gpu], "GPU")
        else:
            # otherwise run on CPU
            tf.config.experimental.set_visible_devices([], "GPU")

        dense_branch = kwargs.get("dense_branch", True)
        conv_branch = kwargs.get("conv_branch", True)
        loss = kwargs.get("loss", "binary_crossentropy")
        optimizer = kwargs.get("optimizer", "adam")
        lr = float(kwargs.get("lr", 3e-4))
        momentum = float(kwargs.get("momentum", 0.9))
        monitor = kwargs.get("monitor", "val_loss")
        patience = float(kwargs.get("patience", 20))
        callbacks = kwargs.get("callbacks", ("reduce_lr_on_plateau", "early_stopping"))

        pre_trained_model = kwargs.get("pre_trained_model")

        classifier = DNN(name=tag)

        classifier.setup(
            dense_branch=dense_branch,
            conv_branch=conv_branch,
            loss=loss,
            optimizer=optimizer,
            lr=lr,
            momentum=momentum,
            monitor=monitor,
            patience=patience,
            callbacks=callbacks,
        )

        if pre_trained_model is not None:
            classifier.model = tf.keras.models.load_model(pre_trained_model)
        if verbose:
            classifier.model.summary()

        if verbose:
            tqdm_callback = tfa.callbacks.TQDMProgressBar()
            classifier.meta["callbacks"].append(tqdm_callback)

        classifier.train(
            datasets["train"],
            datasets["val"],
            steps_per_epoch["train"],
            steps_per_epoch["val"],
            epochs=epochs,
            class_weight=class_weight,
        )

        # eval and save
        stats = classifier.evaluate(
            datasets['test'],
            callbacks=[tfa.callbacks.TQDMProgressBar()],
            verbose=0
        )
        print(stats)

        classifier.save(
            output_path=f"models/{tag}",
            output_format="hdf5",
            tag=f'{datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")}'
        )


if __name__ == "__main__":
    fire.Fire(Scope)
