# Usage

## Download ids for ZTF fields/CCDs/quadrants

- Create CSV file for single CCD/quad pair in a field:

```sh
./get_quad_ids.py --catalog ZTF_source_features_DR5 --field 301 --ccd 2 --quad 3 --minobs 20 --skip 0 --limit 10000
```

- Create multiple HDF5 files for some CCD/quad pairs in a field:

```sh
./get_quad_ids.py --catalog ZTF_source_features_DR5 --field 301 --multi-quads --ccd-range 1 8 --quad-range 2 4 --minobs 20 --limit 10000
```

- Create multiple HDF5 files for all CCD/quad pairs in a field:

```sh
./get_quad_ids.py --catalog ZTF_source_features_DR5 --field 301 --multi-quads --minobs 20 --limit 10000
```

- Create single HDF5 file for all sources in a field:

```sh
./get_quad_ids.py --catalog ZTF_source_features_DR5 --field 301 --whole-field
```

## Training deep learning models

For details on the SCoPe taxonomy and architecture,
please refer to [arxiv:2102.11304](https://arxiv.org/pdf/2102.11304.pdf).

- The training pipeline can be invoked with the `scope.py` utility. For example:

```sh
./scope.py train --tag=vnv --path_dataset=data/training/dataset.d15.csv --batch_size=64 --epochs=100 --verbose=1 --pre_trained_model=models/vnv/20210324_220331/vnv.20210324_220331
```

Refer to `./scope.py train --help` for details.

- All the necessary metadata/configuration could be defined in `config.yaml` under `training`,
but could also be overridden with optional `scope.py train` arguments, e.g.
`./scope.py train ... --batch_size=32 --threshold=0.6 ...`.

- The pipeline uses the `ScopeNet` models defined in `scope.nn` as subclassed `tf.keras.models.Model`'s.
- The `Dataset` class defined in `scope.utils` hides the complexity of our dataset handling "under the rug".
- Datasets and pre-trained models could be fetched from the GCS with the `scope.py` tool:

```sh
./scope.py fetch-datasets
./scope.py fetch-models
```

  This requires permissions to access the `gs://ztf-scope` bucket. Alternatively, you can request access to a Google Drive folder containing the latest trained models [here](https://drive.google.com/drive/folders/1_oLBxveioKtw7LyMJfism745USe9tEGZ?usp=sharing).

- Feature name sets are specified in `config.yaml` under `features`.
  These are referenced in `config.yaml` under `training.classes.<class>.features`.

- Feature stats to be used for feature scaling/standardization before training
  is defined in `config.yaml` under `feature_stats`.

- We use [Weights & Biases](https://wandb.com) to track experiments.
  Project details and access credentials can be defined in `config.yaml` under `wandb`.

An example `bash` script to train all classifier families:

```sh
for class in pnp longt i fla ew eb ea e agn bis blyr ceph dscu lpv mir puls rrlyr rscvn srv wuma yso; \
  do echo $class; \
  for state in 1 2 3 4 5 6 7 8 9 42; \
    do ./scope.py train \
      --tag=$class --path_dataset=data/training/dataset.d15.csv \
      --scale_features=min_max --batch_size=64 \
      --epochs=300 --patience=30 --random_state=$state \
      --verbose=1 --gpu=1 --conv_branch=true --save; \
  done; \
done;
```
## Running inference

* Inference requires the following steps: download ids of a field, download features for all downloaded ids, run inference for all available pre-trained models.
* Requires `models/` folder in the root directory containing the pre-trained models for `dnn` and `xgboost`.
* The commands for inference of the field `<field_number>` (in order)
  ```
  ./tools/get_quad_ids.py --field=<field_number> --whole-field
  ./tools/get_features.py --field=<field_number>
  ./get_all_preds.sh <field_number>
  ```
* Creates a single `.csv` file containing all ids of the field in the rows and inference scores for different classes across the columns.

## Handling different file formats
When our manipulations of `pandas` dataframes is complete, we want to save them in an appropriate file format with the desired metadata. Our code works with multiple formats, each of which have advantages and drawbacks:

- <b>Comma Separated Values (CSV, .csv):</b> in this format, data are plain text and columns are separated by commas. While this format offers a high level of human readability, it also takes more space to store and a longer time to write and read than other formats.

  `pandas` offers the `read_csv()` function and `to_csv()` method to perform I/O operations with this format. Metadata must be included as plain text in the file.

- <b>Hierarchical Data Format (HDF5, .h5):</b> this format stores data in binary form, so it is not human-readable. It takes up less space on disk than CSV files, and it writes/reads faster for numerical data. HDF5 does not serialize data columns containing structures like a `numpy` array, so file size improvements over CSV can be diminished if these structures exist in the data.

  `pandas` includes `read_hdf()` and `to_hdf()` to handle this format, and they require a package like [`PyTables`](https://www.pytables.org/) to work. `pandas` does not currently support the reading and writing of metadata using the above function and method. See `scope/utils.py` for code that handles metadata in HDF5 files.

- <b>Apache Parquet (.parquet):</b> this format stores data in binary form like HDF5, so it is not human-readable. Like HDF5, Parquet also offers significant disk space savings over CSV. Unlike HDF5, Parquet supports structures like `numpy` arrays in data columns.

  While `pandas` offers `read_parquet()` and `to_parquet()` to support this format (requiring e.g. [`PyArrow`](https://arrow.apache.org/docs/python/) to work), these again do not support the reading and writing of metadata associated with the dataframe.  See `scope/utils.py` for code that reads and writes metadata in Parquet files.


## Scope Download Classification
inputs:
1. CSV file containing obj_id and/or ra dec coordinates. Set to "parse" to download sources by group id.
2. target group id(s) on Fritz for download (if CSV file not provided)
3. Index or page number (if in "parse" mode) to begin downloading (optional)
4. Flag to merge features from Kowalski with downloaded sources
5. Name of features catalog to query
6. Limit on number of sources to query at once
7. Filename of classification mapper
8. Name of directory to save downloaded files
9. Name of file containing merged classifications and features
10. Output format of saved files, if not specified in (9). Must be one of .parquet, .h5, or .csv.

process:
1. if CSV file provided, query by object ids or ra, dec
2. if CSV file not provided, bulk query based on group id(s)
3. get the classification/probabilities/periods of the objects in the dataset from Fritz
4. append these values as new columns on the dataset, save to new file
5. if merge_features, query Kowalski and merge sources with features, saving new CSV file
6. To skip the source download part of the code, provide an input CSV file containing columns named 'obj_id', 'classification', 'probability', 'period_origin', 'period', 'ztf_id_origin', and 'ztf_id'.

output: data with new columns appended.

```sh
./scope_download_classification.py -file sample.csv -group_ids 360 361 -start 10 -merge_features True -features_catalog ZTF_source_features_DR5 -features_limit 5000 -mapper_name golden_dataset_mapper.json -output_dir fritzDownload -output_filename merged_classifications_features -output_format .parquet
```

## Scope Upload Classification
inputs:
0. gloria object (include Kowalski host, port, protocol, and token or username+password in config.yaml)
1. CSV file containing ra, dec, period, and labels
2. target group id(s) on Fritz for upload
3. Scope taxonomy id
4. Class name of objects. Set this to "read" and include taxonomy map to automatically upload multiple classes at once.
5. Taxonomy map ("label in file":"Fritz taxonomy name", JSON format).
6. Comment to post (if specified)
7. Index to start uploading (zero-based)
8. Index to stop uploading (inclusive)
9. Skip photometry upload (existing sources only)
10. Origin of ZTF data. If set, values in ztf_id CSV column will post as annotations.

process:
1. get object ids of all the data from Fritz using the ra, dec, and period
2. save the objects to Fritz group
3. in batches, upload the classifications of the objects in the dataset to target group on Fritz
4. duplicate classifications will not be uploaded to Fritz. If n classifications are manually specified, probabilities will be sourced from the last n columns of the dataset.
5. (post comment to each uploaded source)

```sh
./scope_upload_classification.py -file sample.csv -group_ids 500 250 750 -taxonomy_id 7 -classification variable flaring -taxonomy_map map.json -comment vetted -start 35 -stop 50 -skip_phot False -ztf_origin ZTF_DR5
```

## Scope Manage Annotation
inputs:
1. action (one of "post", "update", or "delete")
2. source (ZTF ID or path to .csv file with multiple objects (ID column "obj_id"))
3. target group id(s) on Fritz
4. origin name of annotation
5. key name of annotation
6. value of annotation (required for "post" and "update" - if source is a .csv file, value will auto-populate from source[key])

process:
1. for each source, find existing annotations (for "update" and "delete" actions)
2. interact with API to make desired changes to annotations
3. confirm changes with printed messages

```sh
./scope_manage_annotation.py -action post -source sample.csv -group_ids 200 300 400 -origin revisedperiod -key period
```

## Scope Upload Disagreements
inputs:
1. dataset
2. group id on Fritz
3. gloria object

process:
1. read in the csv dataset to pandas dataframe
2. get high scoring objects on DNN or on XGBoost from Fritz
3. get objects that have high confidence on DNN but low confidence on XGBoost and vice versa
4. get different statistics of those disagreeing objects and combine to a dataframe
5. filter those disagreeing objects that are contained in the training set and remove them
6. upload the remaining disagreeing objects to target group on Fritz

```sh
./scope_upload_disagreements.py -file dataset.d15.csv -id 360 -token sample_token
```
