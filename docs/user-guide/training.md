# Training Models

For details on the SCoPe taxonomy and architecture, please refer to [arxiv:2102.11304](https://arxiv.org/pdf/2102.11304.pdf).

## Configuration

Provide SCoPe your training set's filepath using the `training:` `dataset:` field in `config.yaml`. The path should be a partial one starting within the `scope` directory. For example, if your training set `trainingSet.parquet` is within the `tools` directory (which itself is within `scope`), provide `tools/trainingSet.parquet` in the `dataset:` field.

When running scripts, `scope` will by default use the `config.yaml` file in your current directory. You can specify a different config file by providing its path to any installed script using the `--config-path` argument.

## Training a Classifier

The training pipeline can be invoked with `scope-train`. For example:

```sh
scope-train --tag vnv --algorithm xgb --group ss23 \
  --period-suffix ELS_ECE_EAOV --epochs 30 --verbose --save --plot --skip-cv
```

### Arguments

- `--tag`: the abbreviated name of the classification to train a binary classifier. A list of abbreviations and definitions can be found in the Guide for Fritz Scanners section.
- `--algorithm`: SCoPe currently supports neural network (`dnn`) and XGBoost (`xgb`) algorithms.
- `--group`: if `--save` is passed, training results are saved to the group/directory named here.
- `--period-suffix`: SCoPe determines light curve periods using GPU-accelerated algorithms. These algorithms include a Lomb-Scargle approach (ELS), Conditional Entropy (ECE), Analysis of Variance (AOV), and an approach nesting all three (ELS_ECE_EAOV). Periodic features are stored with the suffix specified here.
- `--min-count`: requires at least `min_count` positive examples to run training.
- `--epochs`: number of training epochs.
- `--path-dataset`: path to the training dataset file.
- `--batch-size`: training batch size.
- `--pre-trained-model`: path to an HDF5 file of pre-trained model weights for continued training.
- `--scale-features`: feature scaling method (e.g. `min_max`).
- `--patience`: early stopping patience (epochs without improvement before stopping).
- `--random-state`: random seed for reproducibility.
- `--gpu`: flag to use GPU for training.
- `--conv-branch`: flag to use convolutional branch.
- `--save`: save the trained model.
- `--plot`: generate diagnostic plots.
- `--skip-cv`: skip cross-validation.

Refer to `scope-train --help` for the full list of options.

### Notes

- All the necessary metadata/configuration can be defined in `config.yaml` under `training`, but can also be overridden with optional `scope-train` arguments, e.g. `scope-train ... --batch-size 32 --threshold 0.6 ...`.
- By default, the pipeline uses the `DNN` models defined in `scope/nn.py` using TensorFlow's `keras` functional API. SCoPe also supports an implementation of XGBoost (set `--algorithm xgb`; see `scope/xgb.py`).
- If `--save` is specified during `DNN` training, an HDF5 file of the model's layers and weights will be saved. This file can be directly used for additional training and inferencing. For `XGB`, a JSON file will save the model along with a `.params` file with the model parameters.
- The `Dataset` class defined in `scope.utils` hides the complexity of dataset handling.
- You can request access to a Google Drive folder containing the latest trained models [here](https://drive.google.com/drive/folders/1_oLBxveioKtw7LyMJfism745USe9tEGZ?usp=sharing).
- Feature name sets are specified in `config.yaml` under `features`. These are referenced in `config.yaml` under `training.classes.<class>.features`.
- Feature stats to be used for feature scaling/standardization before training are either computed by the code (default) or defined in `config.yaml` under `feature_stats`.
- We use [Weights & Biases](https://wandb.com) to track experiments. Project details and access credentials can be defined in `config.yaml` under `wandb`.
- The above XGB training example skips cross-validation in the interest of time. For a full run, remove the `--skip-cv` argument to run a cross-validated grid search of XGB hyperparameters.
- DNN hyperparameters are optimized using [Weights and Biases Sweeps](https://docs.wandb.ai/guides/sweeps). The results of these sweeps are the default hyperparameters in the config file. To run another round of sweeps for DNN, create a WandB account and set the `--run-sweeps` keyword in the call to `scope-train`.
- SCoPe DNN training does not provide feature importance information (due to the hidden layers of the network). Feature importance is possible to estimate for neural networks, but it is more computationally expensive compared to the "free" information from XGB.

## Training Multiple Classifiers

### Using a bash loop

Initially, SCoPe used a `bash` script to train all classifier families:

```sh
for class in pnp longt i fla ew eb ea e agn bis blyr ceph dscu lpv mir puls rrlyr rscvn srv wuma yso; \
  do echo $class; \
  for state in 1 2 3 4 5 6 7 8 9 42; \
    do scope-train \
      --tag $class --path-dataset data/training/dataset.d15.csv \
      --scale-features min_max --batch-size 64 \
      --epochs 300 --patience 30 --random-state $state \
      --verbose 1 --gpu 1 --conv-branch --save; \
  done; \
done;
```

### Using `create-training-script`

A training script containing one line per class can be generated by running `create-training-script`:

```sh
create-training-script --filename train_xgb.sh --min-count 1000 \
  --algorithm xgb --period-suffix ELS_ECE_EAOV \
  --add-keywords "--save --plot --group ss23 --epochs 30 --skip-cv"
```

A path to the training set may be provided as input or otherwise taken from `config.yaml` (`training: dataset:`). To continue training on existing models, specify the `--pre-trained-group-name` keyword containing the models in `create-training-script`. If training on a feature collection containing multiple sets of periodic features (from different algorithms), set the suffix corresponding to the desired algorithm using `--period-suffix` or the `features: info: period_suffix:` field in the config file. The string specified in `--add-keywords` serves as a catch-all for additional keywords to include in each line of the script.

If `--pre-trained-group-name` is specified and the `--train-all` keyword is set, the output script will train all classes specified in `config.yaml` regardless of whether they have a pre-trained model. If `--train-all` is not set (the default), the script will limit training to classes that have an existing trained model.

Modify the permissions of the generated script by running `chmod +x train_xgb.sh`, then run it in a terminal window (e.g. `./train_xgb.sh`) to train multiple classifiers sequentially.

!!! note
    The code will raise an error if the training script filename already exists.

### Running Training on HPC Resources

`train-algorithm-slurm` and `train-algorithm-job-submission` can be used to generate and submit SLURM scripts to train all classifiers in parallel using HPC resources.

## Adding New Features for Training

To add a new feature, first ensure that it has been generated and saved in the training set file. Then, update the config file in the `features:` section. This section contains a list of each feature used by SCoPe. Along with the name of the feature, be sure to specify the boolean `include` value (as `true`), the `dtype`, and whether the feature is `periodic` or not (i.e. whether the code should append a `period_suffix` to the name).

If the new feature is ontological in nature, add the same config info to both the `phenomenological:` and `ontological:` lists. For a phenomenological feature, only add this info to the `phenomenological:` list. Note that changing the config in this way will raise an error when running SCoPe with pre-existing trained models that lack the new feature.

## Plotting Classifier Performance

SCoPe saves diagnostic plots and JSON files to report each classifier's performance. The below code shows the location of the validation set results for one classifier:

```python
import pathlib
import json

path_model = pathlib.Path.home() / "scope/models_xgb/ss23/vnv"
path_stats = [x for x in path_model.glob("*plots/val/*stats.json")][0]

with open(path_stats) as f:
    stats = json.load(f)
```

The code below makes a bar plot of the precision and recall for this classifier:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.rcParams['font.size'] = 13
plt.title("XGB performance (vnv)")
plt.bar("vnv", stats['precision'], color='blue', width=1, label='precision')
plt.bar("vnv", stats['recall'], color='red', width=0.6, label='recall')
plt.legend(ncol=2, loc=0)
plt.ylim(0, 1.15)
plt.xlim(-3, 3)
plt.ylabel('Score')
```

This code may also be placed in a loop over multiple labels to compare each classifier's performance.
