# Usage

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
1. data containing ra, dec, and period
2. gloria object
3. group id on Fritz
4. Fritz token

process:
1. get object ids of all the data from Fritz using the ra, dec, and period
2. save the objects to Fritz group
3. get the classification of the objects in the dataset from Fritz
4. append the classification to a new column on the dataset

output: data with classifcation column appended.

## Scope Upload Classification
inputs:
1. data containing ra, dec, period, and labels
2. gloria object
3. target group id on Fritz for upload
4. Scope taxonomy id
5. Class name of objects
6. Fritz token

process:
1. get object ids of all the data from Fritz using the ra, dec, and period
2. save the objects to Fritz group
3. upload the classification of the objects in the dataset to target group on Fritz

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
