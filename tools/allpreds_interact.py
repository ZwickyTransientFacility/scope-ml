#!/usr/bin/env python
import argparse
import pathlib
import yaml
import glob
import code
import pandas as pd
from scope.utils import read_parquet  # , write_parquet

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

config_path = BASE_DIR / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

default_path_to_preds = config['inference']['path_to_preds']
if default_path_to_preds is None:
    default_path_to_preds = '.'


def allpreds_interact(
    fields=[
        296,
        297,
        423,
        424,
        487,
        488,
        562,
        563,
        682,
        683,
        699,
        700,
        717,
        718,
        777,
        778,
        841,
        842,
        852,
        853,
    ],
    path_to_preds=default_path_to_preds,
):

    path_to_preds = pathlib.Path(path_to_preds)

    dnn_preds = glob.glob(path_to_preds / 'preds_dnn/*')
    xgb_preds = glob.glob(path_to_preds / 'preds_xgb/*')

    preds_dnn = pd.DataFrame([])
    preds_xgb = pd.DataFrame([])

    for d in dnn_preds:
        fieldno = int(d.split('/')[-1].split('_')[-1])
        if fieldno in fields:
            df = read_parquet(f"{d}/field_{fieldno}.parquet")
            preds_dnn = pd.concat([preds_dnn, df])

    for x in xgb_preds:
        fieldno = int(x.split('/')[-1].split('_')[-1])
        if fieldno in fields:
            df = read_parquet(f"{x}/field_{fieldno}.parquet")
            preds_xgb = pd.concat([preds_xgb, df])

    print(
        "DNN/XGB preds available in preds_dnn and preds_xgb variables. Interacting..."
    )

    code.interact(local=locals())

    # Write combined parquet file if specified
    # if write_combined_file:
    #    preds = pd.merge(preds_dnn, preds_xgb, on=['_id','Gaia_EDR3___id','AllWISE___id','PS1_DR1___id','ra','dec','period','field','ccd','quad','filter'])
    #    write_parquet(preds, BASE_DIR / combined_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #    "--write_combined_file",
    #    action='store_true',
    # )
    parser.add_argument(
        "--fields",
        type=list,
        default=[
            296,
            297,
            423,
            424,
            487,
            488,
            562,
            563,
            682,
            683,
            699,
            700,
            717,
            718,
            777,
            778,
            841,
            842,
            852,
            853,
        ],
        help="Fields to include",
    )
    parser.add_argument(
        "--path_to_preds",
        type=str,
        default=default_path_to_preds,
    )
    args = parser.parse_args()

    allpreds_interact(
        fields=args.fields,
        path_to_preds=args.path_to_preds,
    )
