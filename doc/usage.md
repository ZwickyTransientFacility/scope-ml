# Usage

## Download ids for a ZTF quad

```python
get_field_ids.py
ZTF_sources_20210401 -field 301 -ccd 2 -quad 3 -minobs 5 -skip 0 -limit 20 -token sample_token
```

It can then be put in a loop to, say, get 100 ids at a time from a quad,
and/or loop over quads and CCDs to get all ids for a field.

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

  This requires permissions to access the `gs://ztf-scope` bucket.

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

## Scope Download Classification
inputs:
1. CSV file containing obj_id and/or ra dec coordinates. Set to "parse" to download sources by group id.
2. gloria object
3. target group id(s) on Fritz for download (if CSV file not provided)
4. Fritz token

process:
1. if CSV file provided, query by object ids or ra, dec
2. if CSV file not provided, query based on group id(s)
3. get the classification/probabilities/periods of the objects in the dataset from Fritz
4. append these values as new columns on the dataset, save to new file

output: data with new columns appended.

```sh
./scope_download_classification.py -file sample.csv -group_ids 360 361 -token sample_token
```

## Scope Upload Classification
inputs:
0. gloria object (create a file named secrets.json with Kowalski username+password or token, host, port and protocol)
1. CSV file containing ra, dec, period, and labels
2. target group id(s) on Fritz for upload
3. Scope taxonomy id
4. Class name of objects. Set this to "read" and include taxonomy map to automatically upload multiple classes at once.
5. Fritz token
6. Taxonomy map ("label in file":"Fritz taxonomy name", JSON format).
7. Comment to post (if specified)
8. Index to resume uploading (in event of interruption)

process:
1. get object ids of all the data from Fritz using the ra, dec, and period
2. save the objects to Fritz group
3. upload the classification of the objects in the dataset to target group on Fritz
4. duplicate classifications will not be uploaded to Fritz. If n classifications are manually specified, probabilities will be sourced from the last n columns of the dataset.
5. (post comment to each uploaded source)

```sh
./scope_upload_classification.py -file sample.csv -group_ids 500 250 750 -taxonomy_id 7 -classification variable flaring -token sample_token -taxonomy_map map.json -comment vetted -resume 35
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

## Scope Manage Annotation
inputs:
1. action (one of "post", "update", or "delete")
2. source (ZTF ID or path to .csv file with multiple objects (ID column "obj_id"))
3. target group id(s) on Fritz
4. Fritz token
5. origin name of annotation
6. key name of annotation
7. value of annotation (required for "post" and "update" - if source is a .csv file, value will auto-populate from source[key])

process:
1. for each source, find existing annotations (for "update" and "delete" actions)
2. interact with API to make desired changes to annotations
3. confirm changes with printed messages

```sh
./scope_manage_annotation.py -action post -source sample.csv -group_ids 200 300 400 -token sample_token -origin revisedperiod -key period
```
