import fire
import pandas as pd
import glob
import os


def run(ccd: int, quad: int, **kwargs):
    """
    USAGE:  python combine_preds.py --ccd 1 --quad 1 --verbose
    """
    verbose = kwargs.get("verbose", True)

    df_collection = []

    for i, filename in enumerate(
        glob.glob("preds/ccd_" + str(ccd).zfill(2) + f"_quad_{quad}/*.csv")
    ):
        if (not filename.endswith("all_preds.csv")) and (
            not filename.endswith("features.csv")
        ):
            df_temp = pd.read_csv(filename)
            # df_temp = pd.read_pickle(filename)
            if i != 0:
                df_temp.drop("_id", axis=1, inplace=True)
            df_temp.reset_index(inplace=True, drop=True)
            # print(df_temp)
            df_collection += [df_temp]

    df = pd.concat(df_collection, axis=1)
    if verbose:
        print(df)
    os.makedirs("preds/all_preds/", exist_ok=True)

    df.to_csv(
        "preds/ccd_" + str(ccd).zfill(2) + f"_quad_{quad}/all_preds.csv", index=False
    )
    df.to_csv(
        "preds/all_preds/ccd_" + str(ccd).zfill(2) + f"_quad_{quad}.csv", index=False
    )


if __name__ == "__main__":
    fire.Fire(run)
