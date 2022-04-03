# Developer Guidelines

## How to contribute

Contributions to `scope` are made through [GitHub Pull Requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests), a set of proposed commits (or patches).

To prepare, you should:

- Create your own fork the [scope repository](https://github.com/ZwickyTransientFacility/scope) by clicking the "fork" button.

- [Set up SSH authentication with GitHub](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh).

- Clone (download) your copy of the repository, and set up a remote called `upstream` that points to the main `scope` repository.

  ```shell script
  git clone git@github.com:<yourname>/scope && cd scope
  git remote add upstream git@github.com:ZwickyTransientFacility/scope
  ```

Then, for each feature you wish to contribute, create a pull request:

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

3. When ready, push your branch to GitHub:

   ```shell script
   git push origin rewrite-contributor-guide
   ```

   Once the branch is uploaded, GitHub should print a URL for turning your branch into a pull request.
   Open that URL in your browser, write an informative title and description for your pull request, and submit it.

4. The team will now review your contribution, and suggest changes.
*To simplify review, please limit pull requests to one logical set of changes.*
To incorporate changes recommended by the reviewers, commit edits to your branch, and push to the branch again
(there is no need to re-create the pull request, it will automatically track modifications to your branch).

5. Sometimes, while you were working on your feature, the `main` branch is updated with new commits, potentially
resulting in conflicts with your feature branch. The are two ways to resolve this situation - merging and rebasing,
please look [here](https://www.atlassian.com/git/tutorials/merging-vs-rebasing) for a detailed discussion.
While both ways are acceptable, since we are squashing commits from a PR before merging, we prefer the first option:

    ```shell script
    git merge rewrite-contributor-guide upstream/main
    ```
Developers may merge `main` into their branch as many times as they want to.

6. Once the pull request has been reviewed and approved by at least two team members, it will be merged into `scope`.

## Setting up your environment

### Windows Linux MacOS (AMD64)

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

---

You may want to create/activate a virtual environment, for example:

```bash
python -m venv scope-env
source scope-env/bin/activate
```

Make sure the requirements to run it are met, e.g.:

```bash
pip install -r requirements.txt
```

### MacOS(ARM64)

#### Tensorflow for Mac OS M1

You are supposed to install the correct version of Tensorflow. Specifically, it should fit the ARM64 architecture, and the Mac OS based on M1 series CPU.

Apple official provides effective version. At this page `https://developer.apple.com/metal/tensorflow-plugin/` you can gain the methods to finish it.

But there are still some key things.

1. Anaconda doesn't work properly. In its place, You're going to use Miniforge3, also a conda environment, which is specifically adapted to Apple's operating system. Anaconda does not provide the correct version.

2. After you have successfully installed tensorflow-deps, tensorflow-macos, tensorflow-metal, you are going to modify the file `requirements.txt` before you install any other software. You are supposed to remove `tensorflow<2.6` `tensorflow-addons>=0.12` from `.ruquirements/dev.txt` .

   Then, you can use `pip install -r requirements.txt` to install other python packages.

   When you meet an error, you can install it by `conda install xxx` , and remove it from `.ruquirements/dev.txt` . After that, you can use `pip install -r requirements.txt` again.

3. If some packages keep making errors.  You are supposed to update the conda environment.

#### Specific operation

To install the tensorflow for Mac OS 
```zsh
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```

The `.requirements/dev-M1.txt` , please use this file to overwrite the `.requirements/dev.txt` .
```txt
deepdiff>=5.0
gsutil>=4.60
keras-tuner>=1.0.2
matplotlib>=3.3
pytest>=6.1.2
questionary>=1.8.1
scikit-learn>=0.24.1
wandb>=0.12.1
```

You need to install some packages separately.
```zsh
conda install numpy
conda install openblas #to fix the numpy
conda install healpy
conda install pandas
```


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

```{include} ./field_guide__<object_class>.md
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
