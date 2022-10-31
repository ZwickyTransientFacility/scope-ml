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
from typing import Optional, Sequence, Union
import yaml

from scope.utils import forgiving_true, load_config, read_hdf, read_parquet


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


def check_configs(config_wildcards: Sequence = ("config.*yaml",)):
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
            deep_diff = DeepDiff(config_defaults, config_wildcard, ignore_order=True)
            difference = {
                k: v for k, v in deep_diff.items() if k in ("dictionary_item_removed",)
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

    def _get_features(
        self,
        positions: Sequence[Sequence[float]],
        catalog: str = "ZTF_source_features_20210401",
        max_distance: Union[float, int] = 5.0,
        distance_units: str = "arcsec",
    ) -> pd.DataFrame:
        """Get nearest source in feature set for a set of given positions

        :param positions: R.A./Decl. [deg]
        :param catalog: feature catalog to query
        :param max_distance:
        :param distance_units: arcsec | arcmin | deg | rad
        :return:
        """
        if self.kowalski is None:
            raise ConnectionError("Kowalski connection not established.")
        if catalog is None:
            catalog = self.config["kowalski"]["collections"]["features"]
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
                            "period": 1,
                            "ra": 1,
                            "dec": 1,
                        },
                    }
                },
            },
        }
        response = self.kowalski.query(query=query)
        features_nearest = [
            v[0] for k, v in response.get("data").get(catalog).items() if len(v) > 0
        ]
        df = pd.DataFrame.from_records(features_nearest)

        return df

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
                        },
                    }
                },
            },
            "kwargs": {"limit": 1},
        }
        response = self.kowalski.query(query=query)
        gaia_nearest = [
            v[0] for k, v in response.get("data").get(catalog).items() if len(v) > 0
        ]
        df = pd.DataFrame.from_records(gaia_nearest)

        df["M"] = df["phot_g_mean_mag"] + 5 * np.log10(df["parallax"] * 0.001) + 5
        df["Ml"] = (
            df["phot_g_mean_mag"]
            + 5 * np.log10((df["parallax"] + df["parallax_error"]) * 0.001)
            + 5
        )
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
                    "radec": {"target": [ra, dec]},
                },
                "catalogs": {
                    catalog: {
                        "filter": {},
                        "projection": {
                            "_id": 1,
                            "filter": 1,
                            "field": 1,
                            "data.hjd": 1,
                            "data.fid": 1,
                            "data.mag": 1,
                            "data.magerr": 1,
                            "data.ra": 1,
                            "data.dec": 1,
                            "data.programid": 1,
                            "data.catflags": 1,
                        },
                    }
                },
            },
        }
        response = self.kowalski.query(query=query)
        light_curves_raw = response.get("data").get(catalog).get("target")

        light_curves = []
        for light_curve in light_curves_raw:
            df = pd.DataFrame.from_records(light_curve["data"])
            # broadcast to all data points:
            df["_id"] = light_curve["_id"]
            df["filter"] = light_curve["filter"]
            df["field"] = light_curve["field"]
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

        from scope.utils import (
            make_tdtax_taxonomy,
            plot_gaia_density,
            plot_gaia_hr,
            plot_light_curve_data,
            plot_periods,
        )

        # generate taxonomy.html
        with status("Generating taxonomy visualization"):
            path_static = pathlib.Path(__file__).parent.absolute() / "doc" / "_static"
            if not path_static.exists():
                path_static.mkdir(parents=True, exist_ok=True)
            tdtax.write_viz(
                make_tdtax_taxonomy(self.config["taxonomy"]),
                outname=path_static / "taxonomy.html",
            )

        # generate images for the Field Guide
        if (self.kowalski is None) or (not self.kowalski.ping()):
            print("Kowalski connection not established, cannot generate docs.")
            return

        period_limits = {
            "cepheid": [1.0, 100.0],
            "delta_scuti": [0.03, 0.3],
            "beta_lyr": [0.3, 25],
            "rr_lyr": [0.2, 1.0],
            "w_uma": [0.2, 0.8],
        }
        period_loglimits = {
            "cepheid": True,
            "delta_scuti": False,
            "beta_lyr": True,
            "rr_lyr": False,
            "w_uma": False,
        }

        # example periods
        with status("Generating example period histograms"):
            path_doc_data = pathlib.Path(__file__).parent.absolute() / "doc" / "data"

            # stored as ra/decs in csv format under /data/golden
            golden_sets = pathlib.Path(__file__).parent.absolute() / "data" / "golden"
            for golden_set in golden_sets.glob("*.csv"):
                golden_set_name = golden_set.stem
                positions = pd.read_csv(golden_set).to_numpy().tolist()
                features = self._get_features(positions=positions)

                if len(features) == 0:
                    print(f"No features for {golden_set_name}")
                    continue

                limits = period_limits.get(golden_set_name)
                loglimits = period_loglimits.get(golden_set_name)

                plot_periods(
                    features=features,
                    limits=limits,
                    loglimits=loglimits,
                    save=path_doc_data / f"period__{golden_set_name}",
                )

        # example skymaps for all Golden sets
        with status("Generating skymaps diagrams for Golden sets"):
            path_doc_data = pathlib.Path(__file__).parent.absolute() / "doc" / "data"

            path_gaia_density = (
                pathlib.Path(__file__).parent.absolute()
                / "data"
                / "Gaia_hp8_densitymap.fits"
            )
            # stored as ra/decs in csv format under /data/golden
            golden_sets = pathlib.Path(__file__).parent.absolute() / "data" / "golden"
            for golden_set in golden_sets.glob("*.csv"):
                golden_set_name = golden_set.stem
                positions = pd.read_csv(golden_set).to_numpy().tolist()

                plot_gaia_density(
                    positions=positions,
                    path_gaia_density=path_gaia_density,
                    save=path_doc_data / f"radec__{golden_set_name}",
                )

        # example light curves
        with status("Generating example light curves"):
            path_doc_data = pathlib.Path(__file__).parent.absolute() / "doc" / "data"

            for sample_object_name, sample_object in self.config["docs"][
                "field_guide"
            ].items():
                sample_light_curves = self._get_light_curve_data(
                    ra=sample_object["coordinates"][0],
                    dec=sample_object["coordinates"][1],
                    catalog=self.config["kowalski"]["collections"]["sources"],
                )
                plot_light_curve_data(
                    light_curve_data=sample_light_curves,
                    period=sample_object.get("period"),
                    title=sample_object.get("title"),
                    save=path_doc_data / sample_object_name,
                )

        # example HR diagrams for all Golden sets
        with status("Generating HR diagrams for Golden sets"):
            path_gaia_hr_histogram = (
                pathlib.Path(__file__).parent.absolute()
                / "doc"
                / "data"
                / "gaia_hr_histogram.dat"
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
    def fetch_models(gcs_path: str = "gs://ztf-scope/models"):
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
            os.path.join(gcs_path, "*.csv"),
            str(path_models),
        ]
        p = subprocess.run(command, check=True)
        if p.returncode != 0:
            raise RuntimeError("Failed to fetch SCoPe models")

    @staticmethod
    def fetch_datasets(gcs_path: str = "gs://ztf-scope/datasets"):
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
            os.path.join(gcs_path, "*.csv"),
            str(path_datasets),
        ]
        p = subprocess.run(command, check=True)
        if p.returncode != 0:
            raise RuntimeError("Failed to fetch SCoPe datasets")

    def train(
        self,
        tag: str,
        path_dataset: Union[str, pathlib.Path],
        gpu: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Train classifier

        :param tag: classifier designation, refers to "class" in config.taxonomy
        :param path_dataset: local path to .parquet, .h5 or .csv file with the dataset
        :param gpu: GPU id to use, zero-based. check tf.config.list_physical_devices('GPU') for available devices
        :param verbose:
        :param kwargs: refer to utils.DNN.setup and utils.Dataset.make
        :return:
        """

        import tensorflow as tf

        if gpu is not None:
            # specified a GPU to run on?
            gpus = tf.config.list_physical_devices("GPU")
            tf.config.experimental.set_visible_devices(gpus[gpu], "GPU")
        else:
            # otherwise run on CPU
            tf.config.experimental.set_visible_devices([], "GPU")

        import wandb
        from wandb.keras import WandbCallback

        from scope.nn import DNN
        from scope.utils import Dataset

        train_config = self.config["training"]["classes"][tag]

        features = self.config["features"][train_config["features"]]

        ds = Dataset(
            tag=tag,
            path_dataset=path_dataset,
            features=features,
            verbose=verbose,
            **kwargs,
        )

        label = train_config["label"]

        # values from kwargs override those defined in config. if latter is absent, use reasonable default
        threshold = kwargs.get("threshold", train_config.get("threshold", 0.5))
        balance = kwargs.get("balance", train_config.get("balance", None))
        weight_per_class = kwargs.get(
            "weight_per_class", train_config.get("weight_per_class", False)
        )
        scale_features = kwargs.get("scale_features", "min_max")

        test_size = kwargs.get("test_size", train_config.get("test_size", 0.1))
        val_size = kwargs.get("val_size", train_config.get("val_size", 0.1))
        random_state = kwargs.get("random_state", train_config.get("random_state", 42))
        feature_stats = self.config.get("feature_stats", None)

        batch_size = kwargs.get("batch_size", train_config.get("batch_size", 64))
        shuffle_buffer_size = kwargs.get(
            "shuffle_buffer_size", train_config.get("shuffle_buffer_size", 512)
        )
        epochs = kwargs.get("epochs", train_config.get("epochs", 100))

        datasets, indexes, steps_per_epoch, class_weight = ds.make(
            target_label=label,
            threshold=threshold,
            balance=balance,
            weight_per_class=weight_per_class,
            scale_features=scale_features,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            feature_stats=feature_stats,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            epochs=epochs,
        )

        # set up and train model
        dense_branch = kwargs.get("dense_branch", True)
        conv_branch = kwargs.get("conv_branch", True)
        loss = kwargs.get("loss", "binary_crossentropy")
        optimizer = kwargs.get("optimizer", "adam")
        lr = float(kwargs.get("lr", 3e-4))
        momentum = float(kwargs.get("momentum", 0.9))
        monitor = kwargs.get("monitor", "val_loss")
        patience = int(kwargs.get("patience", 20))
        callbacks = kwargs.get("callbacks", ("reduce_lr_on_plateau", "early_stopping"))
        run_eagerly = kwargs.get("run_eagerly", False)
        pre_trained_model = kwargs.get("pre_trained_model")
        save = kwargs.get("save", False)

        # parse boolean args
        dense_branch = forgiving_true(dense_branch)
        conv_branch = forgiving_true(conv_branch)
        run_eagerly = forgiving_true(run_eagerly)
        save = forgiving_true(save)

        classifier = DNN(name=tag)

        classifier.setup(
            dense_branch=dense_branch,
            features_input_shape=(len(features),),
            conv_branch=conv_branch,
            dmdt_input_shape=(26, 26, 1),
            loss=loss,
            optimizer=optimizer,
            learning_rate=lr,
            momentum=momentum,
            monitor=monitor,
            patience=patience,
            callbacks=callbacks,
            run_eagerly=run_eagerly,
        )

        if verbose:
            print(classifier.model.summary())

        if pre_trained_model is not None:
            classifier.load(pre_trained_model)

        time_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if not kwargs.get("test", False):
            wandb.login(key=self.config["wandb"]["token"])
            wandb.init(
                project=self.config["wandb"]["project"],
                tags=[tag],
                name=f"{tag}-{time_tag}",
                config={
                    "tag": tag,
                    "label": label,
                    "dataset": pathlib.Path(path_dataset).name,
                    "scale_features": scale_features,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "patience": patience,
                    "random_state": random_state,
                    "batch_size": batch_size,
                    "architecture": "scope-net",
                    "dense_branch": dense_branch,
                    "conv_branch": conv_branch,
                },
            )
            classifier.meta["callbacks"].append(WandbCallback())

        classifier.train(
            datasets["train"],
            datasets["val"],
            steps_per_epoch["train"],
            steps_per_epoch["val"],
            epochs=epochs,
            class_weight=class_weight,
            verbose=verbose,
        )

        if verbose:
            print("Evaluating on test set:")
        stats = classifier.evaluate(datasets["test"], verbose=verbose)
        if verbose:
            print(stats)

        param_names = (
            "loss",
            "tp",
            "fp",
            "tn",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "auc",
        )
        if not kwargs.get("test", False):
            # log model performance on the test set
            for param, value in zip(param_names, stats):
                wandb.run.summary[f"test_{param}"] = value
            p, r = wandb.run.summary["test_precision"], wandb.run.summary["test_recall"]
            wandb.run.summary["test_f1"] = 2 * p * r / (p + r)

        if datasets["dropped_samples"] is not None:
            # log model performance on the dropped samples
            if verbose:
                print("Evaluating on samples dropped from the training set:")
            stats = classifier.evaluate(datasets["dropped_samples"], verbose=verbose)
            if verbose:
                print(stats)

            if not kwargs.get("test", False):
                for param, value in zip(param_names, stats):
                    wandb.run.summary[f"dropped_samples_{param}"] = value
                p, r = (
                    wandb.run.summary["dropped_samples_precision"],
                    wandb.run.summary["dropped_samples_recall"],
                )
                wandb.run.summary["dropped_samples_f1"] = 2 * p * r / (p + r)

        if save:
            output_path = str(pathlib.Path(__file__).parent.absolute() / "models" / tag)
            if verbose:
                print(f"Saving model to {output_path}")
            classifier.save(
                output_path=output_path,
                output_format="tf",
                tag=time_tag,
            )

            return time_tag

    def create_training_script(
        self,
        filename: str = 'train_dnn.sh',
        min_count: int = 100,
        path_dataset: str = None,
    ):
        """
        Create training shell script from classes in config file meeting minimum count requirement

        :param filename: filename of shell script (must not currently exist) (str)
        :param min_count: minimum number of positive examples to include in script (int)
        :param path_dataset: local path to .parquet, .h5 or .csv file with the dataset, if not provided in config.yaml (str)
        :return:
        """
        path = str(pathlib.Path(__file__).parent.absolute() / filename)
        phenom_keys = []
        ontol_keys = []

        if path_dataset is None:
            dataset_name = self.config['training']['dataset']
            path_dataset = str(pathlib.Path(__file__).parent.absolute() / dataset_name)

        if path_dataset.endswith('.parquet'):
            dataset = read_parquet(path_dataset)
        elif path_dataset.endswith('.h5'):
            dataset = read_hdf(path_dataset)
        elif path_dataset.endswith('.csv'):
            dataset = pd.read_csv(path_dataset)
        else:
            raise ValueError(
                'Dataset in config file must end with .parquet, .h5 or .csv'
            )

        with open(path, 'x') as script:

            script.write('#!/bin/bash\n')

            for key in self.config['training']['classes'].keys():
                label = self.config['training']['classes'][key]['label']
                threshold = self.config['training']['classes'][key]['threshold']
                branch = self.config['training']['classes'][key]['features']
                num_pos = np.sum(dataset[label] > threshold)
                print(
                    f'Label {label}: {num_pos} positive examples with P > {threshold}'
                )

                if num_pos > min_count:
                    if branch == 'phenomenological':
                        phenom_keys += [key]
                    else:
                        ontol_keys += [key]

            script.write('# Phenomenological\n')
            script.writelines(
                [
                    f'./scope.py train --tag {key} --path_dataset {path_dataset} --verbose \n'
                    for key in phenom_keys
                ]
            )
            script.write('# Ontological\n')
            script.writelines(
                [
                    f'./scope.py train --tag {key} --path_dataset {path_dataset} --verbose \n'
                    for key in ontol_keys
                ]
            )

    def test(self):
        """Test different workflows

        :return:
        """
        import uuid
        import shutil

        # create a mock dataset and check that the training pipeline works
        dataset = f"{uuid.uuid4().hex}.csv"
        path_mock = pathlib.Path(__file__).parent.absolute() / "data" / "training"

        try:
            if not path_mock.exists():
                path_mock.mkdir(parents=True, exist_ok=True)

            feature_names = self.config["features"]["ontological"]
            class_names = [
                self.config["training"]["classes"][class_name]["label"]
                for class_name in self.config["training"]["classes"]
            ]

            entries = []
            for i in range(1000):
                entry = {
                    **{
                        feature_name: np.random.normal(0, 0.1)
                        for feature_name in feature_names
                    },
                    **{
                        class_name: np.random.choice([0, 1])
                        for class_name in class_names
                    },
                    **{"non-variable": np.random.choice([0, 1])},
                    **{"dmdt": np.abs(np.random.random((26, 26))).tolist()},
                }
                entries.append(entry)

            df_mock = pd.DataFrame.from_records(entries)
            df_mock.to_csv(path_mock / dataset, index=False)

            tag = "vnv"
            time_tag = self.train(
                tag=tag,
                path_dataset=path_mock / dataset,
                batch_size=32,
                epochs=3,
                verbose=True,
                save=True,
                test=True,
            )
            path_model = (
                pathlib.Path(__file__).parent.absolute() / "models" / tag / time_tag
            )
            shutil.rmtree(path_model)
        finally:
            # clean up after thyself
            (path_mock / dataset).unlink()


if __name__ == "__main__":
    fire.Fire(Scope)
