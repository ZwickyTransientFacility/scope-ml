#!/usr/bin/env python
import pandas as pd
import os
import pathlib
import argparse
from scope.utils import read_parquet, write_parquet

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()


def combine_preds(
    combined_preds_dirname: str = 'preds_dnn_xgb',
    specific_field: str = None,
    save: bool = True,
):
    """
    Combine DNN and XGB preds for ingestion into Kowalski

    :param combined_preds_dirname: directory name to use for combined preds (str)
    :param specific_field: number of specific field to run (str, useful for testing)
    :param save: if True, save combined preds (bool, useful for testing)

    """
    if specific_field is not None:
        glob_input = specific_field
    else:
        glob_input = '*'

    field_paths_dnn = [x for x in (BASE_DIR / 'preds_dnn').glob(f'field_{glob_input}')]
    fields_dnn = [x.name for x in field_paths_dnn]
    fields_dnn_dict = {
        fields_dnn[i]: field_paths_dnn[i] for i in range(len(fields_dnn))
    }

    field_paths_xgb = [x for x in (BASE_DIR / 'preds_xgb').glob(f'field_{glob_input}')]
    fields_xgb = [x.name for x in field_paths_xgb]
    fields_xgb_dict = {
        fields_xgb[i]: field_paths_xgb[i] for i in range(len(fields_xgb))
    }

    if save:
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
            if save:
                write_parquet(
                    combined_preds,
                    BASE_DIR / combined_preds_dirname / f"{field}.parquet",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combined_preds_dirname",
        type=str,
        default='preds_dnn_xgb',
        help="dirname in which to save combined preds",
    )
    parser.add_argument(
        "--specific_field",
        type=str,
        default=None,
        help="specific field to combine preds (useful for testing)",
    )
    parser.add_argument(
        "--doNotSave",
        action='store_true',
        help="if set, do not save results (useful for testing)",
    )
    args = parser.parse_args()

    combine_preds(
        combined_preds_dirname=args.combined_preds_dirname,
        specific_field=args.specific_field,
        save=not args.doNotSave,
    )
