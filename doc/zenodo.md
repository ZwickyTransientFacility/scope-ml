# Data Releases on Zenodo

As more ZTF fields receive SCoPe classifications, we will want to make them public on Zenodo. This page describes the data products available and provides a guide to preparing/publishing new data releases.

## Data product description

The most recent data release contains these components:

| file | description |
| -------------- | ------------ |
| field_296_100rows.csv | Demo predictions file containing 100 classified light curves |
| fields.json | List of fields in classification catalog, generated when running `combine-preds` |
| predictions_dnn_xgb_*N*_fields.zip | Zip file containing classification catalog (contains *N* combined DNN/XGB prediction files in CSV format) |
| SCoPe_classification_demo.ipynb | Notebook interacting with demo predictions |
| trained_dnn_models.zip | Zip file containing trained DNN models |
| trained_xgb_models.zip | Zip file containing trained XGB models |
| training_set.parquet | Parquet file containing training set |

## Preparing a new release

The permanent link for the SCoPe repository on Zenodo is [https://zenodo.org/doi/10.5281/zenodo.8410825](https://zenodo.org/doi/10.5281/zenodo.8410825). This link will always resolve to the latest data release.

To begin a new release draft, click "New version" on the current release. The main difference between releases is the classification catalog, which adds more ZTF fields over time, along with `fields.json`. Other unchanged elements of the release may be imported from the previous release by clicking "Import files".

## Publishing a new release

To publish a new release after uploading the latest classification catalog, first ensure that all desired files are in place. Then, click "Get a DOI now!" to reserve a version-specific DOI for this release. Additionally, specify the publication date and version in `vX.Y.Z` format.

Finally, create release notes by clicking "Add description" and choosing "Notes" under the "Type" dropdown. These release notes should specify what has changed from one version to the next.

## Adding new users

The "ZTF Source Classification Project" community [`ztf-scope`](https://zenodo.org/communities/ztf-scope) contains the current data repository. Inviting additional Zenodo users to this community as Curators will allow them to manage the repository and create new versions.
