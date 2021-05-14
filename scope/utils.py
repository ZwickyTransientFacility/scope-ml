__all__ = [
    "load_config",
    "plot_gaia_hr",
    "plot_light_curve_data",
]

from astropy.io import fits
import healpy as hp
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

    if period is not None:
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    else:
        fig = plt.figure(figsize=(16, 5))
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
    save: Optional[str] = None,
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


def plot_gaia_density(
    positions: pd.DataFrame,
    path_gaia_density: Union[str, pathlib.Path],
    title: Optional[str] = None,
    save: Optional[str] = None,
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
    fig, ax = plt.subplots(figsize=(6, 6))
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

    # galaxy
    for gallat in [15, 0, -15]:
        theta = np.arange(0.0, 360, 0.036)
        phi = gallat * np.ones_like(theta)
        hp.projplot(theta, phi, "w-", coord=["G"], lonlat=True, lw=2)

    # ecliptic
    for ecllat in zip([0, -30, 30], [2, 1, 1]):
        theta = np.arange(0.0, 360, 0.036)
        phi = gallat * np.ones_like(theta)
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
