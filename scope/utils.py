__all__ = [
    "load_config",
    "plot_gaia_hr",
    "plot_light_curve_data",
]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from typing import Optional, Union
import yaml


def load_config(config_path: str):
    """
    Load config and secrets
    """
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    return config


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

    fig = plt.figure(figsize=(16, 9))
    if title is not None:
        fig.suptitle(title, fontsize=24)

    if period is None:
        ax1 = fig.add_subplot(111)
    else:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

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
                    df.loc[mask_filter, "hjd"] / period % 1 + n,
                    df.loc[mask_filter, "mag"],
                    df.loc[mask_filter, "magerr"],
                    marker=".",
                    color=colors[band],
                    lw=0,
                )

            # invert y axes
            for ax in [ax1, ax2]:
                ax.invert_yaxis()

    ax1.set_xlabel("Time")
    ax1.grid(lw=0.3)
    if period is not None:
        ax2.set_xlabel(f"phase [period={period:4.4g} days]")
        ax2.set_xlim(-1, 1)
        ax2.grid(lw=0.3)

    if save is not None:
        fig.tight_layout()
        plt.savefig(save)


def plot_gaia_hr(
    gaia_data: pd.DataFrame,
    path_gaia_hr_histogram: Union[str, pathlib.Path],
    title: Optional[str] = None,
    save: Optional[str] = None,
):
    """Plot the Gaia HR diagram with a sample of objects over-plotted

    source: https://vlas.dev/post/gaia-dr2-hrd/

    """
    # plot the H-R diagram for 1 M stars within 200 pc from the Sun
    plt.rc("text", usetex=True)

    # load background histogram
    histogram = np.loadtxt(path_gaia_hr_histogram)

    # make figure
    fig, ax = plt.subplots(figsize=(6, 6))
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
        gaia_data["M"] - gaia_data["Ml"],
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
