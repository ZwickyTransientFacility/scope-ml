# Installation/Developer Guidelines

## Initial steps

- Create your own fork the [scope repository](https://github.com/ZwickyTransientFacility/scope) by clicking the "fork" button. Then, decide whether you would like to use HTTPS (easier for beginners) or SSH.
- Following one set of instructions below, clone (download) your copy of the repository, and set up a remote called `upstream` that points to the main `scope` repository.

### HTTPS:

```shell script
git clone https://github.com/<yourname>/scope.git && cd scope
git remote add upstream https://github.com/ZwickyTransientFacility/scope.git
```

### SSH:

- [Set up SSH authentication with GitHub](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh).

```shell script
git clone git@github.com:<yourname>/scope.git && cd scope
git remote add upstream git@github.com:ZwickyTransientFacility/scope.git
```

## Setting up your environment (Windows/Linux/macOS)

### Use a package manager for installation

We currently recommend running `scope` with Python 3.11. You may want to begin your installation by creating/activating a virtual environment, for example using conda. We specifically recommend installing miniforge3 (https://github.com/conda-forge/miniforge).

Once you have a package manager installed, run:

```bash
conda create -n scope-env -c conda-forge python=3.11
conda activate scope-env
```

### Update your `PYTHONPATH`

Ensure that Python can import from `scope` by modifying the `PYTHONPATH` environment variable. Use a simple text editor like `nano` to modify the appropriate file (depending on which shell you are using). For example, if using bash, run `nano ~/.bash_profile` and add the following line:

```bash
export PYTHONPATH="$PYTHONPATH:$HOME/scope"
```

Save the updated file (`Ctrl+O` in `nano`) and close/reopen your terminal for this change to be recognized. Then `cd` back into scope and activate your `scope-env` again.

### Install pre-commit

We use `black` to format the code and `flake8` to verify that code complies with [PEP8](https://www.python.org/dev/peps/pep-0008/).
Please install our pre-commit hook as follows:

```shell script
pip install pre-commit
pre-commit install
```

This will check your changes before each commit to ensure that they
conform with our code style standards. We use black to reformat Python
code.

The pre-commit hook will lint *changes* made to the source.

## Install required packages

Install the required python packages by running:
```bash
pip install -r requirements.txt
```

### Create and modify config.yaml

From the included config.defaults.yaml, make a copy called config.yaml:

```bash
cp config.defaults.yaml config.yaml
```

Edit config.yaml to include Kowalski instance and Fritz tokens in the associated empty `token:` fields.

### Testing
Run `./scope.py test` to test your installation. Note that for the test to pass, you will need access to the Kowalski database. If you do not have Kowalski access, you can run `./scope.py test_limited` to run a more limited (but still useful) set of tests.

### Troubleshooting
Upon encountering installation/testing errors, manually install the package in question using  `conda install xxx` , and remove it from `.requirements/dev.txt`. After that, re-run `pip install -r requirements.txt` to continue.

### Known issues
- Across all platforms, we are currently aware of `scope` dependency issues with Python 3.11.
- Anaconda continues to cause problems with environment setup.
- Using `pip` to install `healpy` on an arm64 Mac can raise an error upon import. We recommend including `h5py` as a requirement during the creation of your `conda` environment.
- On Windows machines, `healpy` and `cesium` raise errors upon installation.
   - For `healpy`, see [this](https://healpy.readthedocs.io/en/latest/install.html#installation-on-windows-through-the-windows-subsystem-for-linux) guide for a potential workaround.
   - For `cesium`, try to install from the source (https://cesium-ml.org/docs/install.html#from-source) within `scope`. If you will not be running feature generation, this is not a critical error, but there will be points in the code that fail (e.g. `scope.py test`, `tools/generate_features.py`)

If the installation continues to raise errors, update the conda environment and try again.

## How to contribute

Contributions to `scope` are made through [GitHub Pull Requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests), a set of proposed commits (or patches):

1. Download the latest version of `scope`, and create a new branch for your work.

   Here, let's say we want to contribute some documentation fixes; we'll call our branch `rewrite-contributor-guide`.

   ```shell script
   git checkout main
   git pull upstream main
   git checkout -b rewrite-contributor-guide
   ```

2. Make modifications to `scope` and commit your changes using `git add` and `git commit`.
Each commit message should consist of a summary line and a longer description, e.g.:

   ```text
   Rewrite the contributor guide
   While reading through the contributor guide, I noticed several places
   in which instructions were out of order. I therefore reorganized all
   sections to follow logically, and fixed several grammar mistakes along
   the way.
   ```

1. When ready, push your branch to GitHub:

   ```shell script
   git push origin rewrite-contributor-guide
   ```

   Once the branch is uploaded, GitHub should print a URL for turning your branch into a pull request.
   Open that URL in your browser, write an informative title and description for your pull request, and submit it.

2. The team will now review your contribution, and suggest changes.
*To simplify review, please limit pull requests to one logical set of changes.*
To incorporate changes recommended by the reviewers, commit edits to your branch, and push to the branch again
(there is no need to re-create the pull request, it will automatically track modifications to your branch).

1. Sometimes, while you were working on your feature, the `main` branch is updated with new commits, potentially
resulting in conflicts with your feature branch. The are two ways to resolve this situation - merging and rebasing,
please look [here](https://www.atlassian.com/git/tutorials/merging-vs-rebasing) for a detailed discussion.
While both ways are acceptable, since we are squashing commits from a PR before merging, we prefer the first option:

    ```shell script
    git merge rewrite-contributor-guide upstream/main
    ```
Developers may merge `main` into their branch as many times as they want to.

1. Once the pull request has been reviewed and approved by at least one team member, it will be merged into `scope`.

## Contributing Field Guide sections

If you would like to contribute a Field Guide section, please follow the steps below.

- Make sure to follow the steps described above in the "How to contribute" section!

- Add sections to `config.defaults.yaml` under `docs.field_guide.<object_class_type>`.
  - Use `docs.field_guide.rr_lyr_ab` as an example. You need to specify the object's
    coordinates and a title for the generated light curve plot. Optionally,
    you may specify a period [days] - then a phase-folded light curve will also be rendered.

- Make sure your `config.yaml` file contains a valid Kowalski token.
  - See [here](https://github.com/dmitryduev/penquins) on how to generate one
  (Kowalski account required).
  - You can use `config.defaults.yaml` as a template.

- Make sure the structure of your config file is the same as the default,
  i.e. you propagated the changes in `config.defaults.yaml`.
  (otherwise the `scope.py` utility will later complain and ask you to fix that).

- Add a Markdown file to `doc/` and call it `field_guide__<object_class>.md`.
  - Use [`doc/field_guide__rr_lyrae.md`](field_guide__rr_lyrae.md) as a template.
  - Light curve examples will be generated as `data/<object_class_type>.png` files using the info
    you provided in the config.
  - Add the following include statements to [`doc/field_guide.md`](field_guide.md):

```
{include} ./field_guide__<object_class>.md
```

- If you wish to render a sample Gaia-based HR diagram, you need to create a "Golden" data set
  for that class of objects and put it under `data/golden` as `<object_class>.csv`
  - The `csv` file must follow the same structure as [`data/golden/rr_lyr.csv`].
   Please keep the `csv` header ("ra,dec") and provide object coordinates in degrees.
  - The HR diagram will be generated as `data/hr__<object_class>.png`

- Run the `./scope.py doc` command to generate the imagery and build the documentation.
  - If the build is successful, you can check the results in
    [`doc/_build/html/index.html`](_build/html/index.html)

- Once you're happy with the result, commit the changes to a branch on your fork
  and open a pull request on GitHub (see the "How to contribute" section above).
  - The GitHub Actions CI will run a subset of the testing/deployment pipeline
    for each commit you make to your branch -
    make sure you get a green checkmark next to the commit hash.
  - Once the PR is reviewed, approved, and merged,
    the CI will automatically build and deploy the docs to
    [`https://scope.ztf.dev`](https://scope.ztf.dev)
