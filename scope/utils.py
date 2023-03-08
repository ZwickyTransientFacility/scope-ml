__all__ = [
    "Dataset",
    "forgiving_true",
    "load_config",
    "log",
    "make_tdtax_taxonomy",
    "plot_gaia_density",
    "plot_gaia_hr",
    "plot_light_curve_data",
    "plot_periods",
    "read_hdf",
    "write_hdf",
    "read_parquet",
    "write_parquet",
    "impute_features",
]

from astropy.io import fits
import datetime
import json
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm.auto import tqdm
from typing import Mapping, Optional, Union
import yaml
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import json as JSON
from sklearn.impute import KNNImputer


def load_config(config_path: Union[str, pathlib.Path]):
    """
    Load config and secrets
    """
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    return config


def time_stamp():
    """

    :return: UTC time as a formatted string
    """
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")


def log(message: str):
    print(f"{time_stamp()}: {message}")


def forgiving_true(expression):
    return True if expression in ("t", "True", "true", "1", 1, True) else False


def make_tdtax_taxonomy(taxonomy: Mapping):
    """Recursively convert taxonomy definition from config["taxonomy"]
       into tdtax-parsable dictionary

    :param taxonomy: config["taxonomy"] section
    :return:
    """
    tdtax_taxonomy = dict()
    if taxonomy["class"] not in ("tds", "phenomenological", "ontological"):
        tdtax_taxonomy["name"] = f"{taxonomy['class']}: {taxonomy['name']}"
    else:
        tdtax_taxonomy["name"] = taxonomy["name"]
    if "subclasses" in taxonomy:
        tdtax_taxonomy["children"] = []
        for cls in taxonomy["subclasses"]:
            tdtax_taxonomy["children"].append(make_tdtax_taxonomy(cls))

    return tdtax_taxonomy


def write_hdf(
    dataframe: pd.DataFrame, filepath: str, key: str = 'df', overwrite: bool = True
):
    """
    Write HDF5 file and attach metadata

    :param dataframe: pandas.DataFrame (metadata must be dict in DataFrame.attrs)
    :param filepath: file path to save HDF5 file (str)
    :param key: key associated with DataFrame (str)
    :param overwrite: if True, overwrite file, else append. (bool)
    """
    mode = 'w' if overwrite else 'a'

    with pd.HDFStore(filepath, mode=mode) as store:
        store.put(key, dataframe)
        store.get_storer(key).attrs.metadata = dataframe.attrs


def read_hdf(filepath: str, key: str = 'df'):
    """
    Read HDF5 file and metadata (if available). Currently supports accessing one key of the file at a time.

    :param filepath: path to read HDF5 file (str)
    :param key: key to access in HDF5 file (str)

    :return: pandas.DataFrame
    """
    with pd.HDFStore(filepath, mode='r') as store:
        dataframe = store[key]
        try:
            dataframe.attrs = store.get_storer(key).attrs.metadata
        except AttributeError:
            warnings.warn('Did not read metadata from HDF5 file.')

    return dataframe


def write_parquet(dataframe: pd.DataFrame, filepath: str, meta_key: str = 'scope'):
    """
    Write Apache Parquet file and attach Metadata

    :param dataframe: pandas.DataFrame (metadata must be dict in DataFrame.attrs)
    :param filepath: file path to save parquet file (str)
    :param meta_key: key associated with metadata to save (str)
    """
    # code adapted from https://towardsdatascience.com/saving-metadata-with-dataframes-71f51f558d8e
    # 2022-10-19

    # Create tables
    table = pa.Table.from_pandas(dataframe)
    # Serialize metadata from DataFrame.attrs
    custom_meta_json = JSON.dumps(dataframe.attrs)
    # Get existing metadata
    existing_meta = table.schema.metadata
    # Combine existing and new metadata.
    combined_meta = {
        meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    # Make new table with combined metadata
    table = table.replace_schema_metadata(combined_meta)
    # Write to parquet file
    pq.write_table(table, filepath)


def read_parquet(filepath: str, meta_key: str = 'scope'):
    """
    Read Apache Parquet file and metadata (if available)

    :param filepath: path of parquet file (str)
    :param meta_key: key associated with saved metadata (str)

    :return: pandas.DataFrame
    """
    # code adapted from https://towardsdatascience.com/saving-metadata-with-dataframes-71f51f558d8e
    # 2022-10-19
    table = pq.read_table(filepath)
    dataframe = table.to_pandas()

    try:
        meta_json = table.schema.metadata[meta_key.encode()]
        restored_meta = JSON.loads(meta_json)
        dataframe.attrs = restored_meta
    except KeyError:
        warnings.warn('Did not read metadata from parquet file.')

    return dataframe


def plot_light_curve_data(
    light_curve_data: pd.DataFrame,
    period: Optional[float] = None,
    title: Optional[str] = None,
    save: Optional[str] = None,
):
    """Plot and save to file light curve data

    :param light_curve_data:
    :param period: float [days] if set, a phase-folded light curve will be displayed
    :param title: plot title
    :param save: path to save the plot
    :return:
    """
    plt.close("all")

    # Official start of ZTF MSIP survey, March 17, 2018
    jd_start = 2458194.5

    colors = {
        1: "#28a745",
        2: "#dc3545",
        3: "#00415a",
        "default": "#f3dc11",
    }

    mask_good_data = light_curve_data["catflags"] == 0
    df = light_curve_data.loc[mask_good_data]

    if period is not None:
        fig = plt.figure(figsize=(16, 9), dpi=200)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    else:
        fig = plt.figure(figsize=(16, 5), dpi=200)
        ax1 = fig.add_subplot(111)

    if title is not None:
        fig.suptitle(title, fontsize=24)

    # plot different ZTF bands/filters
    for band in df["filter"].unique():
        mask_filter = df["filter"] == band
        ax1.errorbar(
            df.loc[mask_filter, "hjd"] - jd_start,
            df.loc[mask_filter, "mag"],
            df.loc[mask_filter, "magerr"],
            marker=".",
            color=colors[band],
            lw=0,
        )
        if period is not None:
            for n in [0, -1]:
                ax2.errorbar(
                    (df.loc[mask_filter, "hjd"] - jd_start) / period % 1 + n,
                    df.loc[mask_filter, "mag"],
                    df.loc[mask_filter, "magerr"],
                    marker=".",
                    color=colors[band],
                    lw=0,
                )
    # invert y axes since we are displaying magnitudes
    ax1.invert_yaxis()
    if period is not None:
        ax2.invert_yaxis()

    ax1.set_xlabel("Time")
    ax1.grid(lw=0.3)
    if period is not None:
        ax2.set_xlabel(f"phase [period={period:4.4g} days]")
        ax2.set_xlim(-1, 1)
        ax2.grid(lw=0.3)

    if save is not None:
        fig.tight_layout()
        plt.savefig(save)


def plot_periods(
    features: pd.DataFrame,
    limits: Optional[list] = None,
    loglimits: Optional[bool] = False,
    number_of_bins: Optional[int] = 20,
    title: Optional[str] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
):
    """Plot a histogram of periods for the sample"""
    # plot the H-R diagram for 1 M stars within 200 pc from the Sun
    plt.rc("text", usetex=True)

    # make figure
    fig, ax = plt.subplots(figsize=(6, 6))
    if title is not None:
        fig.suptitle(title, fontsize=24)

    if limits is not None:
        if loglimits:
            edges = np.logspace(
                np.log10(limits[0]), np.log10(limits[1]), number_of_bins
            )
        else:
            edges = np.linspace(limits[0], limits[1], number_of_bins)
    else:
        if loglimits:
            edges = np.linspace(
                np.log10(0.9 * np.min(features["period"])),
                np.log10(1.1 * np.max(features["period"])),
                number_of_bins,
            )
        else:
            edges = np.linspace(
                0.9 * np.min(features["period"]),
                1.1 * np.max(features["period"]),
                number_of_bins,
            )
    hist, bin_edges = np.histogram(features["period"], bins=edges)
    hist = hist / np.sum(hist)
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    ax.plot(bins, hist, linestyle="-", drawstyle="steps")
    ax.set_xlabel("Period [day]")
    ax.set_ylabel("Probability Density Function")

    # display grid behind all other elements on the plot
    ax.set_axisbelow(True)
    ax.grid(lw=0.3)

    if loglimits:
        ax.set_xscale("log")
    ax.set_xlim([0.9 * bins[0], 1.1 * bins[-1]])

    if save is not None:
        fig.tight_layout()
        plt.savefig(save)


def plot_gaia_hr(
    gaia_data: pd.DataFrame,
    path_gaia_hr_histogram: Union[str, pathlib.Path],
    title: Optional[str] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
):
    """Plot the Gaia HR diagram with a sample of objects over-plotted

    source: https://vlas.dev/post/gaia-dr2-hrd/

    """
    # plot the H-R diagram for 1 M stars within 200 pc from the Sun
    plt.rc("text", usetex=True)

    # load background histogram
    histogram = np.loadtxt(path_gaia_hr_histogram)

    # make figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    if title is not None:
        fig.suptitle(title, fontsize=24)

    x_edges = np.arange(-0.681896, 5.04454978, 0.02848978)
    y_edges = np.arange(-2.90934, 16.5665952, 0.0968952)

    ax.pcolormesh(x_edges, y_edges, histogram.T, antialiased=False)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    ax.invert_yaxis()
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$M_G$")

    # plot sample data
    ax.errorbar(
        gaia_data["BP-RP"],
        gaia_data["M"],
        gaia_data["Ml"] - gaia_data["M"],
        marker=".",
        color="#e68a00",
        alpha=0.75,
        ls="",
        lw=0.5,
    )

    # display grid behind all other elements on the plot
    ax.set_axisbelow(True)
    ax.grid(lw=0.3)

    if save is not None:
        fig.tight_layout()
        plt.savefig(save)


def plot_gaia_density(
    positions: pd.DataFrame,
    path_gaia_density: Union[str, pathlib.Path],
    title: Optional[str] = None,
    save: Optional[Union[str, pathlib.Path]] = None,
):
    """Plot the RA/DEC Gaia density plot with a sample of objects over-plotted

    source: https://vlas.dev/post/gaia-dr2-hrd/

    """
    # plot the H-R diagram for 1 M stars within 200 pc from the Sun
    plt.rc("text", usetex=True)

    # load the data
    hdulist = fits.open(path_gaia_density)
    hist = hdulist[1].data["srcdens"][np.argsort(hdulist[1].data["hpx8"])]

    # make figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    if title is not None:
        fig.suptitle(title, fontsize=24)

    # background setup
    coordsys = ["C", "C"]
    nest = True

    # colormap
    cm = plt.cm.get_cmap("viridis")  # colorscale
    cm.set_under("w")
    cm.set_bad("w")

    # plot the data in healpy
    norm = "log"
    hp.mollview(
        hist,
        norm=norm,
        unit="Stars per sq. arcmin.",
        cbar=False,
        nest=nest,
        title="",
        coord=coordsys,
        notext=True,
        cmap=cm,
        flip="astro",
        nlocs=4,
        min=0.1,
        max=300,
    )
    ax = plt.gca()
    image = ax.get_images()[0]
    cbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[0.1, 1, 10, 100],
        fraction=0.15,
        pad=0.05,
        location="bottom",
    )
    cbar.set_label("Stars per sq. arcmin.", size=12)
    cbar.ax.tick_params(labelsize=12)

    ax.tick_params(axis="both", which="major", labelsize=24)

    # borders
    lw = 3
    pi = np.pi
    dtor = pi / 180.0
    theta = np.arange(0, 181) * dtor
    hp.projplot(theta, theta * 0 - pi, "-k", lw=lw, direct=True)
    hp.projplot(theta, theta * 0 + 0.9999 * pi, "-k", lw=lw, direct=True)
    phi = np.arange(-180, 180) * dtor
    hp.projplot(phi * 0 + 1.0e-10, phi, "-k", lw=lw, direct=True)
    hp.projplot(phi * 0 + pi - 1.0e-10, phi, "-k", lw=lw, direct=True)

    # ZTF
    theta = np.arange(0.0, 360, 0.036)
    phi = -30.0 * np.ones_like(theta)
    hp.projplot(theta, phi, "k--", coord=["C"], lonlat=True, lw=2)
    hp.projtext(170.0, -24.0, r"ZTF Limit", lonlat=True)

    theta = np.arange(0.0, 360, 0.036)

    # galaxy
    for gallat in [15, 0, -15]:
        phi = gallat * np.ones_like(theta)
        hp.projplot(theta, phi, "w-", coord=["G"], lonlat=True, lw=2)

    # ecliptic
    for ecllat in [0, -30, 30]:
        phi = ecllat * np.ones_like(theta)
        hp.projplot(theta, phi, "w-", coord=["E"], lonlat=True, lw=2, ls=":")

    # graticule
    hp.graticule(ls="-", alpha=0.1, lw=0.5)

    # labels
    for lat in [60, 30, 0, -30, -60]:
        hp.projtext(360.0, lat, str(lat), lonlat=True)
    for lon in [0, 60, 120, 240, 300]:
        hp.projtext(lon, 0.0, str(lon), lonlat=True)

    # NWES
    plt.text(0.0, 0.5, r"E", ha="right", transform=ax.transAxes, weight="bold")
    plt.text(1.0, 0.5, r"W", ha="left", transform=ax.transAxes, weight="bold")
    plt.text(
        0.5,
        0.992,
        r"N",
        va="bottom",
        ha="center",
        transform=ax.transAxes,
        weight="bold",
    )
    plt.text(
        0.5, 0.0, r"S", va="top", ha="center", transform=ax.transAxes, weight="bold"
    )

    color = "k"
    lw = 10
    alpha = 0.75

    for pos in positions:
        hp.projplot(
            pos[0],
            pos[1],
            color=color,
            markersize=5,
            marker="o",
            coord=coordsys,
            lonlat=True,
            lw=lw,
            alpha=alpha,
            zorder=10,
        )

    if save is not None:
        fig.tight_layout()
        plt.savefig(save)


def impute_features(
    features_df: pd.DataFrame, n_neighbors: int = 5, self_impute: bool = False
):
    # Load config file
    config = load_config(
        pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
    )

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

    # After imputation, drop rows containing NaN features with no imputation strategy
    # (Ensures no subsequent errors due to these missing values)
    feature_list_impute_none = [
        x
        for x in config['features']['ontological']
        if (
            config['features']['ontological'][x]['include']
            and config['features']['ontological'][x]['impute_strategy']
            in ['none', 'None', 'NONE']
        )
    ]

    orig_len = len(features_df)
    features_df = features_df.dropna(subset=feature_list_impute_none).reset_index(
        drop=True
    )
    new_len = len(features_df)
    print()
    print(
        f'Dropped {orig_len - new_len} rows containing missing features with no imputation strategy.'
    )

    return features_df


def get_feature_stats(df: pd.DataFrame, features: list):
    feature_stats = {
        feature: {
            "min": np.min(df[feature]),
            "max": np.max(df[feature]),
            "median": np.median(df[feature]),
            "mean": np.mean(df[feature]),
            "std": np.std(df[feature]),
        }
        for feature in features
    }
    return feature_stats


def overlapping_histogram(a, bins):
    a = a.ravel()
    n = np.zeros(len(bins), int)

    block = 65536
    for i in np.arange(0, len(a), block):
        sa = np.sort(a[i : i + block])
        n += (
            np.r_[
                sa.searchsorted(bins[:-1, 1], 'left'),
                sa.searchsorted(bins[-1, 1], 'right'),
            ]
            - np.r_[
                sa.searchsorted(bins[:-1, 0], 'left'),
                sa.searchsorted(bins[-1, 0], 'right'),
            ]
        )
    return n, (bins[:, 0] + bins[:, 1]) / 2.0


def removeHighCadence(t, m, e, cadence_minutes=30.0):
    idx = []
    for ii in range(len(t)):
        if ii == 0:
            idx.append(ii)
        else:
            dt = t[ii] - t[idx[-1]]
            if dt >= cadence_minutes * 60.0 / 86400.0:
                idx.append(ii)
    if len(idx) > 0:
        t, m, e = t[idx], m[idx], e[idx]
    return t, m, e


def TychoBVfromGaia(G, BP_RP):
    # Conversion formulas from Gaia EDR3 documentation:
    # https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html
    if G < 13:
        Tycho_B = G - (
            -0.004288
            - 0.8547 * (BP_RP)
            + 0.1244 * (BP_RP) ** 2
            - 0.9085 * (BP_RP) ** 3
            + 0.4843 * (BP_RP) ** 4
            - 0.06814 * (BP_RP) ** 5
        )
        Tycho_V = G - (
            -0.01077 - 0.0682 * (BP_RP) - 0.2387 * (BP_RP) ** 2 + 0.02342 * (BP_RP) ** 3
        )
    else:
        Tycho_B, Tycho_V = np.nan, np.nan
    return Tycho_B, Tycho_V


def exclude_radius(Tycho_B, Tycho_V):
    # from Andrew Drake (2022)
    if Tycho_B < 13:
        radius = (
            3.54817e02
            - 5.327823e01 * Tycho_B
            + 6.601137e01 * (Tycho_B - Tycho_V)
            + 2.1031286 * Tycho_B**2
            + 7.6737066 * (Tycho_B - Tycho_V) ** 2
            - 6.164676 * Tycho_B * (Tycho_B - Tycho_V)
        )
    else:
        # For nearby stars fainter than 13th Tycho-B mag, do not need to exclude
        radius = 0.0
    return radius


""" Datasets """


class Dataset(object):
    def __init__(
        self,
        tag: str,
        path_dataset: Union[str, pathlib.Path],
        features: tuple,
        verbose: bool = False,
        **kwargs,
    ):
        """Load parquet, hdf5 or csv file with the dataset containing both data and labels

        :param tag:
        :param path_dataset:
        :param features:
        :param verbose:
        """
        self.tag = tag
        self.path_dataset = str(path_dataset)
        self.features = features
        self.verbose = verbose
        self.target = None

        if self.verbose:
            log(f"Loading {self.path_dataset}...")
        nrows = kwargs.get("nrows", None)

        csv = False
        if self.path_dataset.endswith('.csv'):
            csv = True
            self.df_ds = pd.read_csv(self.path_dataset, nrows=nrows)
        elif self.path_dataset.endswith('.h5'):
            self.df_ds = read_hdf(self.path_dataset)
            for key in ['coordinates', 'dmdt']:
                df_temp = read_hdf(self.path_dataset, key=key)
                self.df_ds[key] = df_temp
            del df_temp
            self.dmdt = self.df_ds['dmdt']
        elif self.path_dataset.endswith('.parquet'):
            self.df_ds = read_parquet(self.path_dataset)
            self.dmdt = self.df_ds['dmdt']
        else:
            raise ValueError('Dataset must have .parquet, .h5 or .csv extension.')

        if self.verbose:
            log(self.df_ds[list(features)].describe())

        self.df_ds = impute_features(self.df_ds, self_impute=True)

        dmdt = []
        if self.verbose:
            print("Moving dmdt's to a dedicated numpy array...")
            iterator = tqdm(self.df_ds.itertuples(), total=len(self.df_ds))
        else:
            iterator = self.df_ds.itertuples()
        for i in iterator:
            data = (
                np.array(json.loads(self.df_ds["dmdt"][i.Index]))
                if csv
                else np.stack(self.df_ds["dmdt"][i.Index])
            )
            if len(data.shape) == 0:
                dmdt.append(np.zeros((26, 26)))
            else:
                dmdt.append(data)

        self.dmdt = np.array(dmdt)
        self.dmdt = np.expand_dims(self.dmdt, axis=-1)

        # drop in df_ds:
        self.df_ds.drop(columns="dmdt")

    @staticmethod
    def threshold(a, t: float = 0.5):
        b = np.zeros_like(a)
        b[np.array(a) > t] = 1
        return b

    def make(
        self,
        target_label: str = "variable",
        threshold: float = 0.5,
        balance: Optional[float] = None,
        weight_per_class: bool = True,
        scale_features: str = "min_max",
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        feature_stats: Optional[dict] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 256,
        epochs: int = 300,
        **kwargs,
    ):
        """Make datasets for target_label

        :param target_label: corresponds to training.classes.<label> in config
        :param threshold: our labels are floats [0, 0.25, 0.5, 0.75, 1]
        :param balance: balance ratio for the prevalent class. if null - use all available data
        :param weight_per_class:
        :param scale_features: min_max | median_std
        :param test_size:
        :param val_size:
        :param random_state: set this for reproducibility
        :param feature_stats: feature_stats to use to standardize features.
                              if None, stats are computed from the data, taking balance into account
        :param batch_size
        :param shuffle_buffer_size
        :param epochs
        :return:
        """

        # Note: Dataset.from_tensor_slices method requires the target variable to be of the int or float32 type.
        float_convert_types = kwargs.get("float_convert_types", (64, 32))
        # TODO: see what to do about it when trying label smoothing in the future.

        target = np.asarray(
            list(map(int, self.threshold(self.df_ds[target_label].values, t=threshold)))
        )

        self.target = np.expand_dims(target, axis=1)

        neg, pos = np.bincount(target.flatten())
        total = neg + pos
        if self.verbose:
            log(
                f"Examples:\n  Total: {total}\n  Positive: {pos} ({100 * pos / total:.2f}% of total)\n"
            )

        w_pos = np.rint(self.df_ds[target_label].values) == 1
        index_pos = self.df_ds.loc[w_pos].index
        if target_label == "variable":
            # 'variable' is a special case: there is an explicit 'non-variable' label:
            w_neg = (
                np.asarray(
                    list(
                        map(
                            int,
                            self.threshold(
                                self.df_ds["non-variable"].values, t=threshold
                            ),
                        )
                    )
                )
                == 1
            )
        else:
            w_neg = ~w_pos
        index_neg = self.df_ds.loc[w_neg].index

        # balance positive and negative examples?
        index_dropped = None
        if balance:
            underrepresented = min(np.sum(w_pos), np.sum(w_neg))
            overrepresented = max(np.sum(w_pos), np.sum(w_neg))
            sample_size = int(min(overrepresented, underrepresented * balance))
            if neg > pos:
                index_neg = (
                    self.df_ds.loc[w_neg].sample(n=sample_size, random_state=1).index
                )
                index_dropped = self.df_ds.loc[
                    list(set(self.df_ds.loc[w_neg].index) - set(index_neg))
                ].index
            else:
                index_pos = (
                    self.df_ds.loc[w_pos].sample(n=sample_size, random_state=1).index
                )
                index_dropped = self.df_ds.loc[
                    list(set(self.df_ds.loc[w_pos].index) - set(index_pos))
                ].index
        if self.verbose:
            log(
                "Number of examples to use in training:"
                f"\n  Positive: {len(index_pos)}\n  Negative: {len(index_neg)}\n"
            )

        ds_indexes = index_pos.to_list() + index_neg.to_list()

        # Train/validation/test split (we will use an 81% / 9% / 10% data split by default):

        train_indexes, test_indexes = train_test_split(
            ds_indexes, shuffle=True, test_size=test_size, random_state=random_state
        )
        train_indexes, val_indexes = train_test_split(
            train_indexes, shuffle=True, test_size=val_size, random_state=random_state
        )

        # Normalize features (dmdt's are already L2-normalized) (?using only the training samples?).
        # Obviously, the same norms will have to be applied at the testing and serving stages.

        # load/compute feature norms:
        if feature_stats is None:
            feature_stats = get_feature_stats(self.df_ds.loc[ds_indexes], self.features)
            if self.verbose:
                print("Computed feature stats:\n", feature_stats)

        # scale features
        for feature in self.features:
            stats = feature_stats.get(feature)
            if (stats is not None) and (stats["std"] != 0):
                if scale_features == "median_std":
                    self.df_ds[feature] = (
                        self.df_ds[feature] - stats["median"]
                    ) / stats["std"]
                elif scale_features == "min_max":
                    self.df_ds[feature] = (self.df_ds[feature] - stats["min"]) / (
                        stats["max"] - stats["min"]
                    )
        # norms = {
        #     feature: np.linalg.norm(self.df_ds.loc[ds_indexes, feature])
        #     for feature in self.features
        # }
        # for feature, norm in norms.items():
        #     if np.isnan(norm) or norm == 0.0:
        #         norms[feature] = 1.0
        # if self.verbose:
        #     print('Computed feature norms:\n', norms)
        #
        # for feature, norm in norms.items():
        #     self.df_ds[feature] /= norm

        # Convert float64 to float32 to satisfy tensorflow requirements
        float_type_dict = {16: np.float16, 32: np.float32, 64: np.float64}
        float_init, float_final = float_convert_types[0], float_convert_types[1]

        # float_init, float_final = float_convert_types[0], float_convert_types[1]

        self.df_ds[
            self.df_ds.select_dtypes(float_type_dict[float_init]).columns
        ] = self.df_ds.select_dtypes(float_type_dict[float_init]).astype(
            float_type_dict[float_final]
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.df_ds.loc[train_indexes, self.features].values,
                    "dmdt": self.dmdt[train_indexes],
                },
                target[train_indexes],
            )
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.df_ds.loc[val_indexes, self.features].values,
                    "dmdt": self.dmdt[val_indexes],
                },
                target[val_indexes],
            )
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "features": self.df_ds.loc[test_indexes, self.features].values,
                    "dmdt": self.dmdt[test_indexes],
                },
                target[test_indexes],
            )
        )
        dropped_samples = (
            tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "features": self.df_ds.loc[index_dropped, self.features].values,
                        "dmdt": self.dmdt[index_dropped],
                    },
                    target[index_dropped],
                )
            )
            if balance
            else None
        )

        # Shuffle and batch the datasets:

        train_dataset = (
            train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat(epochs)
        )
        val_dataset = val_dataset.batch(batch_size).repeat(epochs)
        test_dataset = test_dataset.batch(batch_size)

        dropped_samples = dropped_samples.batch(batch_size) if balance else None

        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "dropped_samples": dropped_samples,
        }

        indexes = {
            "train": np.array(train_indexes),
            "val": np.array(val_indexes),
            "test": np.array(test_indexes),
            "dropped_samples": np.array(index_dropped.to_list())
            if index_dropped is not None
            else None,
        }

        # How many steps per epoch?

        steps_per_epoch_train = len(train_indexes) // batch_size - 1
        steps_per_epoch_val = len(val_indexes) // batch_size - 1
        steps_per_epoch_test = len(test_indexes) // batch_size - 1

        steps_per_epoch = {
            "train": steps_per_epoch_train,
            "val": steps_per_epoch_val,
            "test": steps_per_epoch_test,
        }
        if self.verbose:
            print(f"Steps per epoch: {steps_per_epoch}")

        # Weight training data depending on the number of samples?
        # Very useful for imbalanced classification, especially in the cases with a small number of examples.

        if weight_per_class:
            # weight data class depending on number of examples?
            # num_training_examples_per_class = np.array([len(target) - np.sum(target), np.sum(target)])
            num_training_examples_per_class = np.array([len(index_neg), len(index_pos)])

            assert (
                0 not in num_training_examples_per_class
            ), "found class without any examples!"

            # fewer examples -- larger weight
            weights = (1 / num_training_examples_per_class) / np.linalg.norm(
                (1 / num_training_examples_per_class)
            )
            normalized_weight = weights / np.max(weights)

            class_weight = {i: w for i, w in enumerate(normalized_weight)}

        else:
            # working with binary classifiers only
            class_weight = {i: 1 for i in range(2)}

        return datasets, indexes, steps_per_epoch, class_weight
