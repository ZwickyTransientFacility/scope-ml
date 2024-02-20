#!/usr/bin/env python
import pandas as pd
import os
import pathlib
import argparse
import numpy as np
import json
from scope.utils import read_parquet, write_parquet, parse_load_config

BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

try:
    DEFAULT_PREDS_PATH = pathlib.Path(config['inference']['path_to_preds'])
except TypeError:
    DEFAULT_PREDS_PATH = BASE_DIR

config_fields = config["inference"].get("fields_to_run")


def combine_preds(
    path_to_preds: pathlib.PosixPath = DEFAULT_PREDS_PATH,
    combined_preds_dirname: str = 'preds_dnn_xgb',
    specific_field: str = None,
    use_config_fields: bool = False,
    dateobs: str = None,
    merge_dnn_xgb: bool = False,
    dnn_directory: str = 'preds_dnn',
    xgb_directory: str = 'preds_xgb',
    save: bool = True,
    write_csv: bool = False,
    agg_method: str = 'mean',
    p_threshold: float = 0.7,
):
    """
    Combine DNN and XGB preds for ingestion into Kowalski

    :param path_to_preds: path to directories of existing and combined preds (str)
    :param combined_preds_dirname: directory name to use for combined preds (str)
    :param specific_field: number of specific field to run (str, useful for testing)
    :param use_config_fields: if set, use fields stored in the inference:fields_to_run part of config.yaml (bool
    :param dateobs: GCN dateobs if not running on field/fields (str)
    :param merge_dnn_xgb: if set, combine dnn and xgb classifications instead of keeping separate (bool)
    :param dnn_directory: dirname in which dnn preds are saved (str)
    :param xgb_directory: dirname in which xgb preds are saved (str)
    :param save: if set, save combined preds (bool, useful for testing)
    :param write_csv: if set, save CSV file in addition to parquet (bool)
    :param agg_method: Aggregation method for classification probabilities, 'mean' or 'max' (str)
    :param p_threshold: Minimum probability to add classification to metadata file (float)

    """
    if (specific_field is not None) & (dateobs is not None):
        raise ValueError("Please specify only one of --specific_field and --dateobs.")
    elif (use_config_fields) & (dateobs is not None):
        raise ValueError(
            "Please specify only one of --use_config_fields and --dateobs."
        )

    if isinstance(path_to_preds, str):
        path_to_preds = pathlib.Path(path_to_preds)

    if specific_field is not None:
        glob_input = f"field_{specific_field}"
    elif dateobs is not None:
        if ':' in dateobs:
            dateobs.replace(':', '-')
        glob_input = f"GCN_sources_{dateobs}"
    else:
        glob_input = 'field_*[!specific_ids]'

    field_paths_dnn = [x for x in (path_to_preds / dnn_directory).glob(glob_input)]
    if dateobs is not None:
        fields_dnn = [x.name for x in field_paths_dnn if dateobs in x.name]
    elif not use_config_fields:
        fields_dnn = [x.name for x in field_paths_dnn]
    else:
        field_paths_dnn = [
            x for x in field_paths_dnn if int(x.name.split("_")[-1]) in config_fields
        ]
        fields_dnn = [x.name for x in field_paths_dnn]
    fields_dnn_dict = {
        fields_dnn[i]: field_paths_dnn[i] for i in range(len(fields_dnn))
    }

    field_paths_xgb = [x for x in (path_to_preds / xgb_directory).glob(glob_input)]
    if dateobs is not None:
        fields_xgb = [x.name for x in field_paths_xgb if dateobs in x.name]
    elif not use_config_fields:
        fields_xgb = [x.name for x in field_paths_xgb]
    else:
        field_paths_xgb = [
            x for x in field_paths_xgb if int(x.name.split("_")[-1]) in config_fields
        ]
        fields_xgb = [x.name for x in field_paths_xgb]
    fields_xgb_dict = {
        fields_xgb[i]: field_paths_xgb[i] for i in range(len(fields_xgb))
    }

    if save:
        os.makedirs(path_to_preds / combined_preds_dirname, exist_ok=True)
    counter = 0
    print(f"Processing {len(fields_dnn_dict)} fields/files...")
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
            xgb_columns = [x for x in xgb_preds.columns]

            if not merge_dnn_xgb:
                id_col = '_id' if dateobs is None else 'obj_id'

                dnn_columns.remove(id_col)

                new_xgb_columns = [x for x in xgb_columns if (x not in dnn_columns)]
                xgb_preds_new = xgb_preds[new_xgb_columns]

                preds_to_save = pd.merge(dnn_preds, xgb_preds_new, on=id_col)
                meta_dict = None
            else:
                field = f"merged_{field}"

                merged_preds = pd.merge(dnn_preds, xgb_preds, on='obj_id')
                shared_obj_ids = merged_preds['obj_id'].values

                # Rename e.g. vnv_dnn and vnv_xgb both to vnv
                dnn_rename_mapper = {
                    c: c.split('_')[0] for c in dnn_columns if '_dnn' in c
                }
                xgb_rename_mapper = {
                    c: c.split('_')[0] for c in xgb_columns if '_xgb' in c
                }

                dnn_preds = dnn_preds.rename(dnn_rename_mapper, axis=1)
                xgb_preds = xgb_preds.rename(xgb_rename_mapper, axis=1)

                combined_preds = pd.concat([dnn_preds, xgb_preds])
                combined_columns = [x for x in combined_preds.columns if '_id' not in x]
                pred_columns = np.array(
                    [x for x in combined_columns if x not in ['ra', 'dec', 'period']]
                )

                if agg_method not in ['mean', 'max']:
                    raise ValueError(
                        "Currently supported aggregation methods are 'mean', 'max'."
                    )

                agg_dct = {c: agg_method for c in combined_columns}
                grouped_preds = combined_preds.groupby(['obj_id', 'survey_id'])
                aggregated_preds = grouped_preds.agg(agg_dct)

                preds_to_save = aggregated_preds.loc[shared_obj_ids].reset_index()

                meta_dict = {}
                for _, row in preds_to_save.iterrows():
                    gt_threshold = (row[pred_columns] > p_threshold).values
                    new_entry = {row['obj_id']: (pred_columns[gt_threshold]).tolist()}
                    meta_dict.update(new_entry)

            if save:
                write_parquet(
                    preds_to_save,
                    path_to_preds / combined_preds_dirname / f"{field}.parquet",
                )
                if write_csv:
                    preds_to_save.to_csv(
                        path_to_preds / combined_preds_dirname / f"{field}.csv",
                        index=False,
                    )
                if meta_dict is not None:
                    with open(
                        path_to_preds / combined_preds_dirname / f"{field}_meta.json",
                        'w',
                    ) as f:
                        json.dump(meta_dict, f)

    return preds_to_save


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-preds",
        type=pathlib.PosixPath,
        default=DEFAULT_PREDS_PATH,
        help="path to directories of existing and combined preds",
    )
    parser.add_argument(
        "--combined-preds-dirname",
        type=str,
        default='preds_dnn_xgb',
        help="dirname in which to save combined preds",
    )
    parser.add_argument(
        "--specific-field",
        type=str,
        default=None,
        help="specific field to combine preds (useful for testing)",
    )
    parser.add_argument(
        "--use-config-fields",
        action='store_true',
        help="if set, use fields stored in the inference:fields_to_run part of config.yaml",
    )
    parser.add_argument(
        "--dateobs",
        type=str,
        default=None,
        help="GCN dateobs if not running on field/fields",
    )
    parser.add_argument(
        "--merge-dnn-xgb",
        action='store_true',
        help="if set, combine dnn and xgb classifications instead of keeping separate",
    )
    parser.add_argument(
        "--dnn-directory",
        type=str,
        default='preds_dnn',
        help="dirname in which dnn preds are saved",
    )
    parser.add_argument(
        "--xgb-directory",
        type=str,
        default='preds_xgb',
        help="dirname in which xgb preds preds are saved",
    )
    parser.add_argument(
        "--doNotSave",
        action='store_true',
        help="if set, do not save results (useful for testing)",
    )
    parser.add_argument(
        "--write-csv",
        action='store_true',
        help="if set, save CSV file in addition to parquet",
    )
    parser.add_argument(
        "--agg-method",
        type=str,
        default='mean',
        help="Aggregation method for classification probabilities (mean or max)",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.7,
        help="Minimum probability to add classification to metadata file",
    )

    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    combine_preds(
        path_to_preds=args.path_to_preds,
        combined_preds_dirname=args.combined_preds_dirname,
        specific_field=args.specific_field,
        use_config_fields=args.use_config_fields,
        dateobs=args.dateobs,
        merge_dnn_xgb=args.merge_dnn_xgb,
        dnn_directory=args.dnn_directory,
        xgb_directory=args.xgb_directory,
        save=not args.doNotSave,
        write_csv=args.write_csv,
        agg_method=args.agg_method,
        p_threshold=args.p_threshold,
    )
