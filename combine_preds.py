#!/usr/bin/env python
import pandas as pd
import os
import pathlib
import argparse
from scope.utils import read_parquet, write_parquet

BASE_DIR = pathlib.Path(__file__).parent.absolute()


def combine_preds(
    combined_preds_dirname='preds_dnn_xgb',
):
    """
    Combine DNN and XGB preds for ingestion into Kowalski

    :param combined_preds_dirname: directory name to use for combined preds (str)

    """

    field_paths_dnn = [x for x in (BASE_DIR / 'preds_dnn').glob('field_*')]
    fields_dnn = [x.name for x in field_paths_dnn]
    fields_dnn_dict = {
        fields_dnn[i]: field_paths_dnn[i] for i in range(len(fields_dnn))
    }

    field_paths_xgb = [x for x in (BASE_DIR / 'preds_xgb').glob('field_*')]
    fields_xgb = [x.name for x in field_paths_xgb]
    fields_xgb_dict = {
        fields_xgb[i]: field_paths_xgb[i] for i in range(len(fields_xgb))
    }

    os.makedirs(BASE_DIR / combined_preds_dirname, exist_ok=True)
    counter = 0
    for field in fields_dnn_dict.keys():
        if field in fields_xgb_dict.keys():
            try:
                dnn_preds = read_parquet(fields_dnn_dict[field] / f"{field}.parquet")
                xgb_preds = read_parquet(fields_xgb_dict[field] / f"{field}.parquet")
            except FileNotFoundError:
                print(f'Parquet file not found for field {field}')
                continue

            counter += 1
            dnn_columns = [x for x in dnn_preds.columns]
            dnn_columns.remove('_id')
            xgb_columns = [x for x in xgb_preds.columns]
            new_xgb_columns = [x for x in xgb_columns if (x not in dnn_columns)]
            xgb_preds_new = xgb_preds[new_xgb_columns]

            combined_preds = pd.merge(dnn_preds, xgb_preds_new, on='_id')
            write_parquet(
                combined_preds, BASE_DIR / combined_preds_dirname / f"{field}.parquet"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combined_preds_dirname",
        type=str,
        default='preds_dnn_xgb',
        help="dirname in which to save combined preds",
    )
    args = parser.parse_args()

    combine_preds(
        combined_preds_dirname=args.combined_preds_dirname,
    )
