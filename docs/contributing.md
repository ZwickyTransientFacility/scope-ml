# How to contribute

Contributions to `scope` are made through [GitHub Pull Requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests), a set of proposed commits (or patches):

1. Download the latest version of `scope`, and create a new branch for your work.

   Here, let's say we want to contribute some documentation fixes; we'll call our branch `rewrite-contributor-guide`.

   ```bash
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

   ```bash
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

    ```bash
    git merge rewrite-contributor-guide upstream/main
    ```

   Developers may merge `main` into their branch as many times as they want to.

6. Once the pull request has been reviewed and approved by at least one team member, it will be merged into `scope`.

## Releasing on PyPI

As new features are added to the code, maintainers should occasionally initiate a new release of the `scope-ml` [PyPI](https://pypi.org/project/scope-ml/) package. To do this, first bump the version of the package in `pyproject.toml` and `scope/__init__.py` to the intended `vX.Y.Z` format. Then, navigate to "Releases" in the SCoPe GitHub repo and click "Draft a new release". Enter the version number in "Choose a tag" and click "Generate release notes". It is also advisable to check the box creating a discussion for the release before clicking "Publish release".

Upon release, the `publish-to-pypi.yml` workflow will use GitHub Actions to publish a new version of the package to PyPI automatically. **Note that if the version number has not yet been manually updated in `pyproject.toml`, this workflow will fail.**

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

- Add a Markdown file to `docs/field-guide/` and call it `field_guide__<object_class>.md`.
  - Use `docs/field-guide/field_guide__rr_lyrae.md` as a template.
  - Light curve examples will be generated as `data/<object_class_type>.png` files using the info
    you provided in the config.
  - Add it to the nav in `mkdocs.yml` under the Field Guide section.

- If you wish to render a sample Gaia-based HR diagram, you need to create a "Golden" data set
  for that class of objects and put it under `data/golden` as `<object_class>.csv`
  - The `csv` file must follow the same structure as `data/golden/rr_lyr.csv`.
   Please keep the `csv` header ("ra,dec") and provide object coordinates in degrees.
  - The HR diagram will be generated as `data/hr__<object_class>.png`

- Run the `scope-doc` command to generate the imagery and build the documentation.

- Once you're happy with the result, commit the changes to a branch on your fork
  and open a pull request on GitHub (see the "How to contribute" section above).
  - The GitHub Actions CI will run a subset of the testing/deployment pipeline
    for each commit you make to your branch -
    make sure you get a green checkmark next to the commit hash.
  - Once the PR is reviewed, approved, and merged,
    the CI will automatically build and deploy the docs.
