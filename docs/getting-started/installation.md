# Installation

## Science users

- Create and activate a virtual/conda environment with Python 3.11, e.g:

    ```bash
    conda create -n scope-env python=3.11
    conda activate scope-env
    ```

- Install the latest release of `scope-ml` from PyPI:

    ```bash
    pip install scope-ml
    ```

- In the directory of your choice, run the initialization script. This will create the required directories and copy the necessary files to run the code:

    ```bash
    scope-initialize
    ```

- For accelerated period finding, install [periodfind](https://github.com/ZwickyTransientFacility/periodfind). The GPU (CUDA) backend requires `nvcc`; the CPU (Rust) backend requires `maturin`:

    ```bash
    # GPU backend
    pip install cython numpy && pip install -e .
    # CPU backend
    cd rust && maturin develop --release
    ```

- Change directories to `scope` and modify `config.yaml` to finish the initialization process. This config file is used by default when running all scripts. You can also specify another config file using the `--config-path` argument.

## Developers/contributors

- Create your own fork of the [scope repository](https://github.com/ZwickyTransientFacility/scope) by clicking the "fork" button. Then, decide whether you would like to use HTTPS (easier for beginners) or SSH.
- Following one set of instructions below, clone (download) your copy of the repository, and set up a remote called `upstream` that points to the main `scope` repository.

### HTTPS

```bash
git clone https://github.com/<yourname>/scope.git && cd scope
git remote add upstream https://github.com/ZwickyTransientFacility/scope.git
```

### SSH

- [Set up SSH authentication with GitHub](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh).

```bash
git clone git@github.com:<yourname>/scope.git && cd scope
git remote add upstream git@github.com:ZwickyTransientFacility/scope.git
```

### Setting up your environment (Windows/Linux/macOS)

#### Use a package manager for installation

We currently recommend running `scope` with Python 3.11. You may want to begin your installation by creating/activating a virtual environment, for example using conda. We specifically recommend installing [miniforge3](https://github.com/conda-forge/miniforge).

Once you have a package manager installed, run:

```bash
conda create -n scope-env -c conda-forge python=3.11
conda activate scope-env
```

#### (Optional) Update your `PYTHONPATH`

If you plan to import from `scope`, ensure that Python can import from `scope` by modifying the `PYTHONPATH` environment variable. Use a simple text editor like `nano` to modify the appropriate file (depending on which shell you are using). For example, if using bash, run `nano ~/.bash_profile` and add the following line:

```bash
export PYTHONPATH="$PYTHONPATH:$HOME/scope"
```

Save the updated file (`Ctrl+O` in `nano`) and close/reopen your terminal for this change to be recognized. Then `cd` back into scope and activate your `scope-env` again.

### Install required packages

Ensure you are in the `scope` directory that contains `pyproject.toml`. Then, install the required python packages by running:

```bash
pip install .
```

#### Install dev requirements and pre-commit hook

We use `black` to format the code and `flake8` to verify that code complies with [PEP8](https://www.python.org/dev/peps/pep-0008/). Please install our dev requirements and pre-commit hook as follows:

```bash
pip install -r dev-requirements.txt
pre-commit install
```

This will check your changes before each commit to ensure that they conform with our code style standards. We use black to reformat Python code.

The pre-commit hook will lint *changes* made to the source.

#### Create and modify config.yaml

From the included `config.defaults.yaml`, make a copy called `config.yaml`:

```bash
cp config.defaults.yaml config.yaml
```

Edit `config.yaml` to include Kowalski instance and Fritz tokens in the associated empty `token:` fields.

#### (Optional) Install `periodfind`

For accelerated period finding, install [periodfind](https://github.com/ZwickyTransientFacility/periodfind). The GPU (CUDA) backend requires `nvcc`; the CPU (Rust) backend requires `maturin`:

```bash
# GPU backend
pip install cython numpy && pip install -e .
# CPU backend
cd rust && maturin develop --release
```

#### Testing

Run `scope-test` to test your installation. Note that for the test to pass, you will need access to the Kowalski database. If you do not have Kowalski access, you can run `scope-test-limited` to run a more limited (but still useful) set of tests.

### Troubleshooting

Upon encountering installation/testing errors, manually install the package in question using `conda install xxx`, and remove it from `.requirements/dev.txt`. After that, re-run `pip install -r requirements.txt` to continue.

#### Known issues

- Across all platforms, we are currently aware of `scope` dependency issues with Python 3.12.
- Anaconda may cause problems with environment setup.
- Using `pip` to install `healpy` on an arm64 Mac can raise an error upon import. We recommend including `h5py` as a requirement during the creation of your `conda` environment.
- On Windows machines, `healpy` and `cesium` raise errors upon installation.
    - For `healpy`, see [this guide](https://healpy.readthedocs.io/en/latest/install.html#installation-on-windows-through-the-windows-subsystem-for-linux) for a potential workaround.
    - For `cesium`, try to install from the [source](https://cesium-ml.org/docs/install.html#from-source) within `scope`. If you will not be running feature generation, this is not a critical error, but there will be points in the code that fail (e.g. `scope.py test`, `tools/generate_features.py`).

If the installation continues to raise errors, update the conda environment and try again.
