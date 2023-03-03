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
from scope.utils import (
    forgiving_true,
    load_config,
    read_hdf,
    read_parquet,
    write_hdf,
)
from scope.fritz import radec_to_iau_name
import json


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
            kowalski_alt_token_env = os.environ.get("KOWALSKI_ALT_TOKEN")
            if (kowalski_token_env is not None) & (kowalski_alt_token_env is not None):
                self.config["kowalski"]["token"] = kowalski_token_env
                self.config["kowalski"]["alt_token"] = kowalski_alt_token_env

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
        catalog: str = "ZTF_sources_20210401",
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

    def _drop_low_probs(self, ser, class_list, threshold):
        if ser.name in class_list:
            new_values = [0.0 if v < threshold else v for v in ser.values]
            return new_values
        else:
            return ser

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
                    # catalog=self.config["kowalski"]["collections"]["sources"],
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
        job_type: str = 'train',
        group: str = 'experiment',
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

        all_features = self.config["features"][train_config["features"]]
        features = [
            key for key in all_features if forgiving_true(all_features[key]["include"])
        ]

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
        feature_stats = kwargs.get("feature_stats", None)
        if feature_stats == 'config':
            feature_stats = self.config.get("feature_stats", None)

        batch_size = kwargs.get("batch_size", train_config.get("batch_size", 64))
        shuffle_buffer_size = kwargs.get(
            "shuffle_buffer_size", train_config.get("shuffle_buffer_size", 512)
        )
        epochs = kwargs.get("epochs", train_config.get("epochs", 100))
        float_convert_types = kwargs.get("float_convert_types", (64, 32))

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
            float_convert_types=float_convert_types,
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
        weights_only = kwargs.get("weights_only", False)

        # parse boolean args
        dense_branch = forgiving_true(dense_branch)
        conv_branch = forgiving_true(conv_branch)
        run_eagerly = forgiving_true(run_eagerly)
        save = forgiving_true(save)

        classifier = DNN(name=tag)

        if pre_trained_model is not None:
            classifier.load(pre_trained_model, weights_only=weights_only)
            model_input = classifier.model.input
            training_set_inputs = datasets['train'].element_spec[0]
            # Compare input shapes with model inputs
            print(
                'Comparing shapes of input features with inputs for existing model...'
            )
            for inpt in model_input:
                inpt_name = inpt.name
                inpt_shape = inpt.shape
                inpt_shape.assert_is_compatible_with(
                    training_set_inputs[inpt_name].shape
                )
            print('Input shapes are consistent.')
            classifier.set_callbacks(callbacks, tag, **kwargs)

        else:
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

        time_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if not kwargs.get("test", False):
            wandb.login(key=self.config["wandb"]["token"])
            wandb.init(
                job_type=job_type,
                project=self.config["wandb"]["project"],
                tags=[tag],
                group=group,
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
            output_path = str(
                pathlib.Path(__file__).parent.absolute() / "models" / group / tag
            )
            if verbose:
                print(f"Saving model to {output_path}")
            classifier.save(
                output_path=output_path,
                output_format="h5",
                tag=time_tag,
            )

            return time_tag

    def create_training_script(
        self,
        filename: str = 'train_dnn.sh',
        min_count: int = 100,
        path_dataset: str = None,
        pre_trained_group_name: str = None,
        add_keywords: str = '',
        train_all: bool = False,
    ):
        """
        Create training shell script from classes in config file meeting minimum count requirement

        :param filename: filename of shell script (must not currently exist) (str)
        :param min_count: minimum number of positive examples to include in script (int)
        :param path_dataset: local path to .parquet, .h5 or .csv file with the dataset, if not provided in config.yaml (str)
        :param pre_trained_group_name: name of group containing pre-trained models within models directory (str)
        :param add_keywords: str containing additional training keywords to append to each line in the script
        :param train_all: if group_name is specified, set this keyword to train all classes regardeless of whether a trained model exists (bool)

        :return:

        :examples:  ./scope.py create_training_script --filename='train_dnn.sh' --min_count=1000 \
                    --path_dataset='tools/fritzDownload/merged_classifications_features.parquet' --add_keywords='--save'

                    ./scope.py create_training_script --filename='train_dnn.sh' --min_count=100 \
                    --add_keywords='--save --batch_size=32'
        """
        path = str(pathlib.Path(__file__).parent.absolute() / filename)

        phenom_tags = []
        ontol_tags = []

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

            for tag in self.config['training']['classes'].keys():
                label = self.config['training']['classes'][tag]['label']
                threshold = self.config['training']['classes'][tag]['threshold']
                branch = self.config['training']['classes'][tag]['features']
                num_pos = np.sum(dataset[label] > threshold)

                if num_pos > min_count:
                    print(
                        f'Label {label}: {num_pos} positive examples with P > {threshold}'
                    )
                    if branch == 'phenomenological':
                        phenom_tags += [tag]
                    else:
                        ontol_tags += [tag]

            if pre_trained_group_name is not None:
                group_path = (
                    pathlib.Path(__file__).parent.absolute()
                    / 'models'
                    / pre_trained_group_name
                )
                gen = os.walk(group_path)
                model_tags = [tag[1] for tag in gen]
                model_tags = model_tags[0]

                # If a group name for trained models is provided, either train all classes in config or only those with existing trained model
                phenom_hasmodel = list(
                    set.intersection(set(phenom_tags), set(model_tags))
                )
                ontol_hasmodel = list(
                    set.intersection(set(ontol_tags), set(model_tags))
                )

                script.write('# Phenomenological\n')
                for tag in phenom_tags:
                    if tag in phenom_hasmodel:
                        tag_file_gen = (group_path / tag).glob('*.h5')
                        most_recent_file = max(
                            [file for file in tag_file_gen], key=os.path.getctime
                        ).name

                        script.writelines(
                            f'./scope.py train --tag {tag} --path_dataset {path_dataset} --pre_trained_model models/{pre_trained_group_name}/{tag}/{most_recent_file} --verbose {add_keywords} \n'
                        )

                    elif train_all:
                        script.writelines(
                            f'./scope.py train --tag {tag} --path_dataset {path_dataset} --verbose {add_keywords} \n'
                        )

                script.write('# Ontological\n')
                for tag in ontol_tags:
                    if tag in ontol_hasmodel:
                        tag_file_gen = (group_path / tag).glob('*.h5')
                        most_recent_file = max(
                            [file for file in tag_file_gen], key=os.path.getctime
                        ).name

                        script.writelines(
                            f'./scope.py train --tag {tag} --path_dataset {path_dataset} --pre_trained_model models/{pre_trained_group_name}/{tag}/{most_recent_file} --verbose {add_keywords} \n'
                        )

                    elif train_all:
                        script.writelines(
                            f'./scope.py train --tag {tag} --path_dataset {path_dataset} --verbose {add_keywords} \n'
                        )

            else:
                script.write('# Phenomenological\n')
                script.writelines(
                    [
                        f'./scope.py train --tag {tag} --path_dataset {path_dataset} --verbose {add_keywords} \n'
                        for tag in phenom_tags
                    ]
                )
                script.write('# Ontological\n')
                script.writelines(
                    [
                        f'./scope.py train --tag {tag} --path_dataset {path_dataset} --verbose {add_keywords} \n'
                        for tag in ontol_tags
                    ]
                )

    def create_inference_script(
        self,
        filename: str = 'get_all_preds_dnn.sh',
        group_name: str = 'experiment',
        algorithm: str = 'dnn',
        scale_features: str = 'min_max',
        write_csv: bool = False,
    ):
        """
        Create inference shell script

        :param filename: filename of shell script (must not currently exist) (str)
        :param group_name: name of group containing trained models within models directory (str)
        :param algorithm: algorithm to use in script (str)
        :param scale_features: method to scale features (str, currently "min_max" or "median_std")
        :param write_csv: if True, write CSV file in addition to HDF5 (bool)

        :return:

        :example:  ./scope.py create_inference_script --filename='get_all_preds_dnn.sh' --group_name='experiment' \
                    --algorithm='dnn'
        """

        path = str(pathlib.Path(__file__).parent.absolute() / filename)
        group_path = pathlib.Path(__file__).parent.absolute() / 'models' / group_name

        addtl_args = ''
        if write_csv:
            addtl_args += '--write_csv'

        gen = os.walk(group_path)
        model_tags = [tag[1] for tag in gen]
        model_tags = model_tags[0]

        with open(path, 'x') as script:
            script.write('#!/bin/bash\n')
            script.write(
                '# Call script followed by field number, e.g: ./get_all_preds_dnn.sh 301\n'
            )

            if algorithm in ['dnn', 'DNN', 'nn', 'NN']:
                script.write('echo "dnn inference"\n')
                # Select most recent model for each tag
                for tag in model_tags:
                    tag_file_gen = (group_path / tag).glob('*.h5')
                    most_recent_file = max(
                        [file for file in tag_file_gen], key=os.path.getctime
                    ).name
                    script.write(
                        f'echo -n "{tag} ..." && python tools/inference.py --path-model=models/{group_name}/{tag}/{most_recent_file} --model-class={tag} --field=$1 --whole-field --flag_ids --scale_features={scale_features} {addtl_args} && echo "done"\n'
                    )

            elif algorithm in ['xgb', 'XGB', 'xgboost', 'XGBOOST', 'XGBoost']:
                script.write('echo "xgb inference"\n')
                for tag in model_tags:
                    tag_file_gen = (group_path / tag).glob('*.h5')
                    most_recent_file = max(
                        [file for file in tag_file_gen], key=os.path.getctime
                    ).name
                    script.write(
                        f'echo -n "{tag} ..." && python tools/inference.py --path-model=models/{group_name}/{tag}/{most_recent_file} --model-class={tag} --xgb_model --field=$1 --whole-field --flag_ids {addtl_args} && echo "done"\n'
                    )

            else:
                raise ValueError('algorithm must be DNN or XGB')

    def consolidate_inference_results(
        self,
        dataset: pd.DataFrame,
        statistic: str = 'mean',
    ):
        """
        Consolidate inference results from multiple rows to one per source (called in select_al_sample)

        :param dataset: inference results from 'preds' directory (pandas DataFrame)
        :param statistic: method to combine multiple predictions for single source [mean, median and max currently supported] (str)

        """

        # Define subsets of data with or without Gaia, AllWISE and PS1 IDs.
        # Survey IDs are used to identify unique sources.
        # Begin with Gaia EDR3 ID
        # If no Gaia ID, use AllWISE
        # If no AllWISE, use PS1
        withGaiaID = dataset[dataset['Gaia_EDR3___id'] != 0].reset_index(drop=True)
        nanGaiaID = dataset[dataset['Gaia_EDR3___id'] == 0].reset_index(drop=True)

        withAllWiseID = nanGaiaID[nanGaiaID['AllWISE___id'] != 0].reset_index(drop=True)
        nanAllWiseID = nanGaiaID[nanGaiaID['AllWISE___id'] == 0].reset_index(drop=True)

        withPS1ID = nanAllWiseID[nanAllWiseID['PS1_DR1___id'] != 0].reset_index(
            drop=True
        )

        # Define columns for each subset that should not be averaged or otherwise aggregated
        skip_mean_cols_Gaia = withGaiaID[
            ['Gaia_EDR3___id', 'AllWISE___id', 'PS1_DR1___id', '_id']
        ]
        skip_mean_cols_AllWise = withAllWiseID[
            ['Gaia_EDR3___id', 'AllWISE___id', 'PS1_DR1___id', '_id']
        ]
        skip_mean_cols_PS1 = withPS1ID[
            ['Gaia_EDR3___id', 'AllWISE___id', 'PS1_DR1___id', '_id']
        ]

        if statistic in [
            'mean',
            'Mean',
            'MEAN',
            'average',
            'AVERAGE',
            'Average',
            'avg',
            'AVG',
        ]:
            groupedMeans_Gaia = (
                withGaiaID.groupby('Gaia_EDR3___id')
                .mean()
                .drop(['_id', 'AllWISE___id', 'PS1_DR1___id'], axis=1)
                .reset_index()
            )

            groupedMeans_AllWise = (
                withAllWiseID.groupby('AllWISE___id')
                .mean()
                .drop(['_id', 'Gaia_EDR3___id', 'PS1_DR1___id'], axis=1)
                .reset_index()
            )

            groupedMeans_PS1 = (
                withPS1ID.groupby('PS1_DR1___id')
                .mean()
                .drop(['_id', 'Gaia_EDR3___id', 'AllWISE___id'], axis=1)
                .reset_index()
            )

        elif statistic in ['max', 'Max', 'MAX', 'maximum', 'Maximum', 'MAXIMUM']:
            groupedMeans_Gaia = (
                withGaiaID.groupby('Gaia_EDR3___id')
                .max()
                .drop(['_id', 'AllWISE___id', 'PS1_DR1___id'], axis=1)
                .reset_index()
            )

            groupedMeans_AllWise = (
                withAllWiseID.groupby('AllWISE___id')
                .max()
                .drop(['_id', 'Gaia_EDR3___id', 'PS1_DR1___id'], axis=1)
                .reset_index()
            )

            groupedMeans_PS1 = (
                withPS1ID.groupby('PS1_DR1___id')
                .max()
                .drop(['_id', 'Gaia_EDR3___id', 'AllWISE___id'], axis=1)
                .reset_index()
            )

        elif statistic in ['median', 'Median', 'MEDIAN', 'med', 'MED']:
            groupedMeans_Gaia = (
                withGaiaID.groupby('Gaia_EDR3___id')
                .median()
                .drop(['_id', 'AllWISE___id', 'PS1_DR1___id'], axis=1)
                .reset_index()
            )

            groupedMeans_AllWise = (
                withAllWiseID.groupby('AllWISE___id')
                .median()
                .drop(['_id', 'Gaia_EDR3___id', 'PS1_DR1___id'], axis=1)
                .reset_index()
            )

            groupedMeans_PS1 = (
                withPS1ID.groupby('PS1_DR1___id')
                .median()
                .drop(['_id', 'Gaia_EDR3___id', 'AllWISE___id'], axis=1)
                .reset_index()
            )

        else:
            raise ValueError(
                'Mean, median and max are the currently supported statistics.'
            )

        # Construct new survey_id column that contains the ID used to add grouped source to the list
        string_ids_Gaia = groupedMeans_Gaia['Gaia_EDR3___id'].astype(str)
        groupedMeans_Gaia['survey_id'] = ["Gaia_EDR3___" + s for s in string_ids_Gaia]

        string_ids_AllWise = groupedMeans_AllWise['AllWISE___id'].astype(str)
        groupedMeans_AllWise['survey_id'] = [
            "AllWISE___" + s for s in string_ids_AllWise
        ]

        string_ids_PS1 = groupedMeans_PS1['PS1_DR1___id'].astype(str)
        groupedMeans_PS1['survey_id'] = ["PS1_DR1___" + s for s in string_ids_PS1]

        # Merge averaged, non-averaged columns on obj_id
        allRows_Gaia = pd.merge(
            groupedMeans_Gaia, skip_mean_cols_Gaia, on=['Gaia_EDR3___id']
        )
        groupedMeans_Gaia.drop('Gaia_EDR3___id', axis=1, inplace=True)

        allRows_AllWise = pd.merge(
            groupedMeans_AllWise, skip_mean_cols_AllWise, on=['AllWISE___id']
        )
        groupedMeans_AllWise.drop('AllWISE___id', axis=1, inplace=True)

        allRows_PS1 = pd.merge(
            groupedMeans_PS1, skip_mean_cols_PS1, on=['PS1_DR1___id']
        )
        groupedMeans_PS1.drop('PS1_DR1___id', axis=1, inplace=True)

        # Create dataframe with one row per source
        consol_rows = pd.concat(
            [groupedMeans_Gaia, groupedMeans_AllWise, groupedMeans_PS1]
        ).reset_index(drop=True)

        # Create dataframe containing all rows (including duplicates for multiple light curves)
        all_rows = pd.concat([allRows_Gaia, allRows_AllWise, allRows_PS1])
        all_rows.drop(
            ['Gaia_EDR3___id', 'AllWISE___id', 'PS1_DR1___id'], axis=1, inplace=True
        )

        # Reorder columns for better legibility
        consol_rows = consol_rows.set_index('survey_id').reset_index()
        all_rows = all_rows.set_index('survey_id').reset_index()

        return consol_rows, all_rows

    def select_al_sample(
        self,
        fields: Union[list, str] = 'all',
        group: str = 'experiment',
        min_class_examples: int = 1000,
        select_top_n: bool = False,
        include_all_highprob_labels: bool = False,
        probability_threshold: float = 0.9,
        al_directory: str = 'AL_datasets',
        al_filename: str = 'active_learning_set',
        algorithm: str = 'dnn',
        exclude_training_sources: bool = False,
        write_csv: bool = True,
        verbose: bool = False,
        consolidation_statistic: str = 'mean',
        read_consolidation_results: bool = False,
        write_consolidation_results: bool = False,
        consol_filename: str = 'inference_results',
    ):
        """
        Select subset of predictions to use for active learning.

        :param fields: list of field predictions (integers) to include, or 'all' to use all available fields (list or str)
            note: do not use spaces if providing a list of comma-separated integers to this argument.
        :param group: name of group containing trained models within models directory (str)
        :param min_class_examples: minimum number of examples to include for each class. Some classes may contain fewer than this if the sample is limited (int)
        :param select_top_n: if True, select top N probabilities above probability_threshold from each class (bool)
        :param include_all_highprob_labels: if select_top_n is set, setting this keyword includes any classification above the probability_threshold for all top N sources.
            Otherwise, literally only the top N probabilities for each classification will be included, which may artifically exclude relevant labels.
        :param probability_threshold: minimum probability to select examples for active learning (float)
        :param al_directory: name of directory to create/populate with active learning sample (str)
        :param al_filename: name of file (no extension) to store active learning sample (str)
        :param algorithm: algorithm [dnn or xgb] (str)
        :param exclude_training_sources: if True, exclude sources in current training set from AL sample (bool)
        :param write_csv: if True, write CSV file in addition to HDF5 (bool)
        :param verbose: if True, print additional information (bool)
        :param consolidation_statistic: method to combine multiple classification probabilities for a single source [mean, median or max currently supported] (str)
        :param read_consolidation_results: if True, search for and read an existing consolidated file having _consol.h5 suffix (bool)
        :param write_consolidation_results: if True, save two files: consolidated inference results [1 row per source] and full results [≥ 1 row per source] (bool)
        :param consol_filename: name of file (no extension) to store consolidated and full results (str)

        :return:

        :examples:  ./scope.py select_al_sample --fields=[296,297] --group='experiment' --min_class_examples=1000 --probability_threshold=0.9 --exclude_training_sources --write_consolidation_results
                    ./scope.py select_al_sample --fields=[296,297] --group='experiment' --min_class_examples=500 --select_top_n --include_all_highprob_labels --probability_threshold=0.7 --exclude_training_sources --read_consolidation_results
        """
        base_path = pathlib.Path(__file__).parent.absolute()
        preds_path = base_path / 'preds'

        # Strip extension from filename if provided
        al_filename = al_filename.split('.')[0]
        AL_directory_path = str(base_path / f'{al_directory}_{algorithm}' / al_filename)
        os.makedirs(AL_directory_path, exist_ok=True)

        if algorithm in ['dnn', 'DNN']:
            algorithm = 'dnn'
        elif algorithm in ['xgb', 'XGB']:
            algorithm = 'xgb'
        else:
            raise ValueError('Algorithm must be either dnn or xgb.')

        df_coll = []
        df_coll_allRows = []
        if fields in ['all', 'All', 'ALL']:
            gen_fields = os.walk(preds_path)
            fields = [x for x in gen_fields][0][1]
        else:
            fields = [f'field_{f}' for f in fields]

        print(f'Generating active learning sample from {len(fields)} fields:')

        column_nums = []

        AL_directory_PL = pathlib.Path(AL_directory_path)
        gen = AL_directory_PL.glob(f'{consol_filename}_consol.h5')
        existing_consol_files = [str(x) for x in gen]

        if (read_consolidation_results) & (len(existing_consol_files) > 0):
            print('Loading existing consolidated results...')
            preds_df = read_hdf(existing_consol_files[0])

        else:
            print('Consolidating classification probabilities to one per source...')
            for field in fields:
                print(field)
                h = read_hdf(str(preds_path / field / f'{field}.h5'))
                consolidated_df, all_rows_df = self.consolidate_inference_results(
                    h, statistic=consolidation_statistic
                )

                column_nums += [len(consolidated_df.columns)]
                df_coll += [consolidated_df]
                df_coll_allRows += [all_rows_df]

                if verbose:
                    print(field)
                    print(consolidated_df)
                    print()

                if len(np.unique(column_nums)) > 1:
                    raise ValueError(
                        'Not all predictions have the same number of columns.'
                    )

                # Create consolidated dataframe (one row per source)
                preds_df = pd.concat(df_coll, axis=0)
                # One more groupby to combine sources across multiple fields
                preds_df = preds_df.groupby('survey_id').mean().reset_index()

                # Create dataframe including all light curves (multiple rows per source)
                preds_df_allRows = pd.concat(df_coll_allRows, axis=0)

                # Generate position-based obj_ids for Fritz
                raArr = [ra for ra in preds_df['ra']]
                decArr = [dec for dec in preds_df['dec']]
                obj_ids = [radec_to_iau_name(x, y) for x, y in zip(raArr, decArr)]
                preds_df['obj_id'] = obj_ids

                # Assign obj_ids to all rows
                preds_df_allRows = pd.merge(
                    preds_df_allRows, preds_df[['obj_id', 'survey_id']], on='survey_id'
                )

                # Drop sources which are so close that they cannot be resolved by our position-based ID (~0.0004 of sources)
                preds_df_allRows = (
                    preds_df_allRows.set_index('obj_id')
                    .drop(preds_df[preds_df.duplicated('obj_id')]['obj_id'])
                    .reset_index()
                )
                preds_df = preds_df.drop_duplicates('obj_id', keep=False).reset_index(
                    drop=True
                )

                # Save results
                if write_consolidation_results:
                    write_hdf(
                        preds_df, f'{AL_directory_path}/{consol_filename}_consol.h5'
                    )
                    write_hdf(
                        preds_df_allRows,
                        f'{AL_directory_path}/{consol_filename}_full.h5',
                    )
                    if write_csv:
                        preds_df.to_csv(
                            f'{AL_directory_path}/{consol_filename}_consol.csv',
                            index=False,
                        )
                        preds_df_allRows.to_csv(
                            f'{AL_directory_path}/{consol_filename}_full.csv',
                            index=False,
                        )

        # Define non-variable class as 1 - variable
        include_nonvar = False
        if f'vnv_{algorithm}' in preds_df.columns:
            include_nonvar = True
            preds_df[f'nonvar_{algorithm}'] = np.round(
                1 - preds_df[f'vnv_{algorithm}'], 2
            )

        if exclude_training_sources:
            # Get training set from config file
            training_set_config = self.config['training']['dataset']
            training_set_path = str(base_path / training_set_config)

            if training_set_path.endswith('.parquet'):
                training_set = read_parquet(training_set_path)
            elif training_set_path.endswith('.h5'):
                training_set = read_hdf(training_set_path)
            elif training_set_path.endswith('.csv'):
                training_set = pd.read_csv(training_set_path)
            else:
                raise ValueError(
                    "Training set must be in .parquet, .h5 or .csv format."
                )

            intersec = set.intersection(
                set(preds_df['obj_id'].values), set(training_set['obj_id'].values)
            )
            print(f'Dropping {len(intersec)} sources already in training set...')
            preds_df = preds_df.set_index('obj_id').drop(list(intersec)).reset_index()

        # Use trained model names to establish classes to train
        gen = os.walk(base_path / 'models' / group)
        model_tags = [tag[1] for tag in gen]
        model_tags = model_tags[0]
        model_tags = np.array(model_tags)
        if include_nonvar:
            model_tags = np.concatenate([model_tags, ['nonvar']])

        print(f'Selecting AL sample for {len(model_tags)} classes...')

        toPost_df = pd.DataFrame(columns=preds_df.columns)
        completed_dict = {}
        preds_df.set_index('obj_id', inplace=True)
        toPost_df.set_index('obj_id', inplace=True)

        # Fix random state to allow reproducible results
        rng = np.random.RandomState(9)

        if not select_top_n:
            for tag in model_tags:
                # Idenfity all sources above probability threshold
                highprob_preds = preds_df[
                    preds_df[f'{tag}_{algorithm}'].values >= probability_threshold
                ]
                # Find existing sources in AL sample above probability threshold
                existing_df = toPost_df[
                    toPost_df[f'{tag}_{algorithm}'].values >= probability_threshold
                ]
                existing_count = len(existing_df)

                # Determine number of sources needed to reach at least min_class_examples
                still_to_post_count = min_class_examples - existing_count

                if still_to_post_count > 0:
                    if len(highprob_preds) >= still_to_post_count:
                        # Randomly select from remaining examples without replacement
                        highprob_toPost = rng.choice(
                            highprob_preds.drop(existing_df.index).index,
                            still_to_post_count,
                            replace=False,
                        )
                        concat_toPost_df = highprob_preds.loc[highprob_toPost]
                    else:
                        # If sample is limited, use all remaining available examples
                        concat_toPost_df = highprob_preds

                toPost_df = pd.concat([toPost_df, concat_toPost_df], axis=0)
                toPost_df.drop_duplicates(keep='first', inplace=True)

        else:
            # Select top N classifications above probability threshold for all classes
            print(
                f'Selecting top {min_class_examples} classifications above P = {probability_threshold}...'
            )
            preds_df.reset_index(inplace=True)
            topN_df = pd.DataFrame()
            class_list = [f'{t}_{algorithm}' for t in model_tags]

            for tag in model_tags:
                goodprob_preds = preds_df[
                    preds_df[f'{tag}_{algorithm}'].values >= probability_threshold
                ]

                if not include_all_highprob_labels:
                    # Return only the top N probabilities for each class, even if other high-probability classifications are excluded
                    topN_preds = (
                        goodprob_preds[
                            [
                                'obj_id',
                                'survey_id',
                                'ra',
                                'dec',
                                'period',
                                f'{tag}_{algorithm}',
                            ]
                        ]
                        .sort_values(by=f'{tag}_{algorithm}', ascending=False)
                        .iloc[:min_class_examples]
                        .reset_index(drop=True)
                    )

                else:
                    # Include not only the top N probabilities for each class but also any other classifications above probability_threshold for these sources
                    topN_preds = (
                        goodprob_preds.sort_values(
                            by=f'{tag}_{algorithm}', ascending=False
                        )
                        .iloc[:min_class_examples]
                        .reset_index(drop=True)
                    )

                    topN_preds = topN_preds.apply(
                        self._drop_low_probs,
                        class_list=class_list,
                        threshold=probability_threshold,
                    )

                topN_df = pd.concat([topN_df, topN_preds]).reset_index(drop=True)

            toPost_df = topN_df.fillna(0.0).groupby('obj_id').max().reset_index()

        for tag in model_tags:
            # Make metadata dictionary of example count per class
            completed_dict[f'{tag}_{algorithm}'] = int(
                np.sum(toPost_df[f'{tag}_{algorithm}'].values >= probability_threshold)
            )

        final_toPost = toPost_df.reset_index(drop=True)

        # Write hdf5 and csv files
        write_hdf(final_toPost, f'{AL_directory_path}/{al_filename}.h5')
        if write_csv:
            final_toPost.to_csv(f'{AL_directory_path}/{al_filename}.csv', index=False)

        # Write metadata
        meta_filepath = f'{AL_directory_path}/meta.json'
        with open(meta_filepath, "w") as f:
            try:
                json.dump(completed_dict, f)  # dump dictionary to a json file
            except Exception as e:
                print("error dumping to json, message: ", e)

    def test(self):
        """Test different workflows

        :return:
        """
        import uuid
        from tools import generate_features

        # Test feature generation
        test_field, test_ccd, test_quad = 297, 2, 2
        test_feature_directory = 'generated_features'
        test_feature_filename = 'testFeatures'
        n_sources = 3

        _ = generate_features.generate_features(
            period_algorithm='LS',
            doCPU=True,
            field=test_field,
            ccd=test_ccd,
            quad=test_quad,
            dirname=test_feature_directory,
            filename=test_feature_filename,
            limit=n_sources,
            doCesium=True,
            stop_early=True,
        )

        path_gen_features = (
            pathlib.Path(__file__).parent.absolute()
            / test_feature_directory
            / f"field_{test_field}"
            / f"{test_feature_filename}_field_{test_field}_ccd_{test_ccd}_quad_{test_quad}.parquet"
        )
        os.remove(path_gen_features)

        # create a mock dataset and check that the training pipeline works
        dataset = f"{uuid.uuid4().hex}.csv"
        path_mock = pathlib.Path(__file__).parent.absolute() / "data" / "training"
        group_mock = 'experiment'

        try:
            if not path_mock.exists():
                path_mock.mkdir(parents=True, exist_ok=True)

            all_feature_names = self.config["features"]["ontological"]
            feature_names = [
                key
                for key in all_feature_names
                if forgiving_true(all_feature_names[key]['include'])
            ]
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
                pathlib.Path(__file__).parent.absolute()
                / "models"
                / group_mock
                / tag
                / f"{tag}.{time_tag}.h5"
            )
            os.remove(path_model)
        finally:
            # clean up after thyself
            (path_mock / dataset).unlink()


if __name__ == "__main__":
    fire.Fire(Scope)
