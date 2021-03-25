# Usage

## Training deep learning models

For details on the SCoPe taxonomy and architecture,
please refer to [arxiv:2102.11304](https://arxiv.org/pdf/2102.11304.pdf).

- The training pipeline can be invoked with the `scope.py` utility. For example:

```sh
./scope.py train --tag=vnv --path_dataset=data/training/dataset.d13.csv --batch_size=64 --epochs=100 --verbose=1 --pre_trained_model=models/vnv/20210324_220331/vnv.20210324_220331
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
