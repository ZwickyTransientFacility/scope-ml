#!/usr/bin/env python
from contextlib import contextmanager
from deepdiff import DeepDiff
import fire
import os
import pathlib
from penquins import Kowalski
from pprint import pprint
import questionary
import subprocess
import sys
from typing import Sequence
import yaml

from scope.utils import load_config


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
            kowalski_token_env = os.getenv("KOWALSKI_TOKEN")
            if kowalski_token_env is not None:
                self.config["kowalski"]["token"] = kowalski_token_env

        with status("Setting up Kowalski connection"):
            self.kowalski = Kowalski(
                token=self.config["kowalski"]["token"],
                protocol=self.config["kowalski"]["protocol"],
                host=self.config["kowalski"]["host"],
                port=self.config["kowalski"]["port"],
            )
            if not self.kowalski.ping():
                raise ConnectionError("Could not connect to Kowalski.")

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
        subprocess.run(["make", "html"], cwd="doc", check=True)
        print(self.kowalski.ping())


if __name__ == "__main__":
    fire.Fire(Scope)
