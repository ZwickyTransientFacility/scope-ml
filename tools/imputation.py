import pandas as pd
import numpy as np
from scope.utils import load_config, read_hdf, read_parquet
from sklearn.impute import KNNImputer
import pathlib

# Load config file
config = load_config(pathlib.Path(__file__).parent.parent.absolute() / "config.yaml")


def impute_features(
    features_df: pd.DataFrame, n_neighbors: int = 5, self_impute: bool = False
):

    if self_impute:
        referenceSet = features_df.copy()
    else:
        # Load training set
        trainingSetPath = config['training']['dataset']
        if trainingSetPath.endswith('.parquet'):
            trainingSet = read_parquet(trainingSetPath)
        elif trainingSetPath.endswith('.h5'):
            trainingSet = read_hdf(trainingSetPath)
        elif trainingSetPath.endswith('.csv'):
            trainingSet = pd.read_csv(trainingSetPath)
        else:
            raise ValueError(
                'Training set must have one of .parquet, .h5 or .csv file formats.'
            )

        referenceSet = trainingSet

    # Impute zero where specified
    feature_list_impute_zero = [
        x
        for x in config['features']['ontological']
        if (
            config['features']['ontological'][x]['include']
            and config['features']['ontological'][x]['impute_strategy']
            in ['zero', 'Zero', 'ZERO']
        )
    ]

    print('Imputing zero for the following features: ', feature_list_impute_zero)
    print()
    for feat in feature_list_impute_zero:
        features_df[feat] = features_df[feat].fillna(0.0)

    # Impute median from reference set where specified
    feature_list_impute_median = [
        x
        for x in config['features']['ontological']
        if (
            config['features']['ontological'][x]['include']
            and config['features']['ontological'][x]['impute_strategy']
            in ['median', 'Median', 'MEDIAN']
        )
    ]

    print('Imputing median for the following features: ', feature_list_impute_median)
    print()
    for feat in feature_list_impute_median:
        features_df[feat] = features_df[feat].fillna(np.nanmedian(referenceSet[feat]))

    # Impute mean from reference set where specified
    feature_list_impute_mean = [
        x
        for x in config['features']['ontological']
        if (
            config['features']['ontological'][x]['include']
            and config['features']['ontological'][x]['impute_strategy']
            in ['mean', 'Mean', 'MEAN']
        )
    ]

    print('Imputing mean for the following features: ', feature_list_impute_mean)
    print()
    for feat in feature_list_impute_mean:
        features_df[feat] = features_df[feat].fillna(np.nanmean(referenceSet[feat]))

    # Impute via regression where specified
    feature_list_regression = [
        x
        for x in config['features']['ontological']
        if (
            config['features']['ontological'][x]['include']
            and config['features']['ontological'][x]['impute_strategy']
            in ['regress', 'Regress', 'REGRESS']
        )
    ]

    print('Imputing by regression on the following features: ', feature_list_regression)
    print()

    # Fit KNNImputer to training set
    imp = KNNImputer(n_neighbors=n_neighbors)
    imp.set_output(transform='pandas')

    fit_feats = imp.fit(referenceSet[feature_list_regression])
    imputed_feats = fit_feats.transform(features_df[feature_list_regression])

    for feat in feature_list_regression:
        features_df[feat] = imputed_feats[feat]

    return features_df
