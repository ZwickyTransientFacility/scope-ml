# Quick Start Guide

This guide is intended to facilitate quick interactions with SCoPe code after you have completed the **Installation/Developer Guidelines** section. More detailed usage info can be found in the **Usage** section. **All of the following examples assume that SCoPe is installed in your home directory. If the `scope` directory is located elsewhere, adjust the example code as necessary.**

## Modify `config.yaml`
To start out, provide SCoPe your training set's filepath using the `training:` `dataset:` field in `config.yaml`. The path should be a partial one starting within the `scope` directory. For example, if your training set `trainingSet.parquet` is within the `tools` directory (which itself is within `scope`), provide `tools/trainingSet.parquet` in the `dataset:` field.

## Training

Train an XGBoost binary classifier using the following code:

```
./scope.py train --tag=vnv --algorithm=xgb --group=ss23 --period_suffix=ELS_ECE_EAOV --epochs=30 --verbose --save --plot --skip_cv
```

### Arguments:
`--tag`: the abbreviated name of the classification to train a binary classifier. A list of abbreviations and definitions can be found in the **Guide for Fritz Scanners** section.

`--algorithm`: SCoPe currently supports neural network (dnn) and XGBoost (xgb) algorithms.

`--group`: if `--save` is passed, training results are saved to the group/directory named here.

`--period_suffix`: SCoPe determines light curve periods using GPU-accelerated algorithms. These algorithms include a Lomb-Scargle approach (ELS), Conditional Entropy (ECE), Analysis of Variance (AOV), and an approach nesting all three (ELS_ECE_EAOV). Periodic features are stored with the suffix specified here.

`--min_count`: requires at least min_count positive examples to run training.

`--epochs`: neural network training takes an --epochs argument that is set to 30 here.

***Notes:***
- *The above training runs the XGB algorithm by default and skips cross-validation in the interest of time. For a full run, you can remove the `--skip_cv` argument to run a cross-validated grid search of XGB hyperparameters during training.*

- *DNN hyperparameters are optimized using a different approach - Weights and Biases Sweeps (https://docs.wandb.ai/guides/sweeps). The results of these sweeps are the default hyperparameters in the config file. To run another round of sweeps for DNN, create a WandB account and set the `--run_sweeps` keyword in the call to `scope.py train`.*

- *SCoPe DNN training does not provide feature importance information (due to the hidden layers of the network). Feature importance is possible to estimate for neural networks, but it is more computationally expensive compared to this "free" information from XGB.*

### Train multiple classifiers with one script

Create a shell script that contains multiple calls to `scope.py train`:
```
./scope.py create_training_script --filename=train_xgb.sh --min_count=1000 --algorithm=xgb --period_suffix=ELS_ECE_EAOV --add_keywords="--save --plot --group=ss23 --epochs=30 --skip_cv"
```

Modify the permissions of this script by running `chmod +x train_xgb.sh`. Run the generated training script in a terminal window (using e.g. `./train_xgb.sh`) to train multiple label sequentially.

***Note:***
- *The code will throw an error if the training script filename already exists.*

### Running training on HPC resources

`train_algorithm_slurm.py` and `train_algorithm_job_submission.py` can be used generate and submit `slurm` scripts to train all classifiers in parallel using HPC resources.

## Plotting Classifier Performance
SCoPe saves diagnostic plots and json files to report each classifier's performance. The below code shows the location of the validation set results for one classifier.

```
import pathlib
import json

path_model = pathlib.Path.home() / "scope/models_xgb/ss23/vnv"
path_stats = [x for x in path_model.glob("*plots/val/*stats.json")][0]

with open(path_stats) as f:
    stats = json.load(f)
```

The code below makes a bar plot of the precision and recall for this classifier:
```
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.rcParams['font.size']=13
plt.title(f"XGB performance (vnv)")
plt.bar("vnv", stats['precision'], color='blue',width=1,label='precision')
plt.bar("vnv", stats['recall'], color='red',width=0.6, label='recall')
plt.legend(ncol=2,loc=0)
plt.ylim(0,1.15)
plt.xlim(-3,3)
plt.ylabel('Score')

```
This code may also be placed in a loop over multiple labels to compare each classifier's performance.

## Inference

Use `tools/inference.py` to run inference on a field (297) of features (within a directory called `generated_features`). The classifiers used for this inference are within the `ss23` directory/group specified during training.

```
./scope.py create_inference_script --filename=get_all_preds_xgb.sh --group_name=ss23 --algorithm=xgb --period_suffix=ELS_ECE_EAOV --feature_directory=generated_features
```

Modify the permissions of this script using `chmod +x get_all_preds_xgb.sh`, then run on the desired field:
```
./get_all_preds_xgb.sh 297
```

***Notes:***
- *`scope.py create_inference_script` will throw an error if the inference script filename already exists.*
- *Inference begins by imputing missing features using the strategies specified in the `features:` section of the config file.*

### Running inference on HPC resources

`run_inference_slurm.py` and `run_inference_job_submission.py` can be used generate and submit `slurm` scripts to run inference for all classifiers in parallel using HPC resources.*

## Examining predictions

The result of running the inference script will be a parquet file containing some descriptive columns followed by columns containing for each classification's probability for each source in the field. By default, the file is located as follows:

```
path_preds = pathlib.Path.home() / "scope/preds_xgb/field_297/field_297.parquet"
```

SCoPe's `read_parquet` utility offers an easy way to read the predictions file and provide it as a `pandas` DataFrame.

```
from scope.utils import read_parquet
preds = read_parquet(path_preds)
```
