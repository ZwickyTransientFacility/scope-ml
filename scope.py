#!/usr/bin/env python
import fire
import subprocess
import sys


class Scope:

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

    @staticmethod
    def doc():
        """Build docs"""
        subprocess.run(["make", "html"], cwd="doc", check=True)


if __name__ == "__main__":
    fire.Fire(Scope)
