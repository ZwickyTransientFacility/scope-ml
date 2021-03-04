__all__ = [
    "load_config",
    "plot_light_curve_data",
]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
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
    plt.close("all")

    # Official start of ZTF MSIP survey, March 1, 2018
    jd_start = 2458178.5

    colors = {
        1: "#28a745",
        2: "#dc3545",
        3: "#00415a",
        "default": "#f3dc11",
    }

    mask_good_data = light_curve_data["catflags"] == 0
    df = light_curve_data.loc[mask_good_data]

    fig = plt.figure(figsize=(16, 9))
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
            marker='.',
            color=colors[band],
            lw=0
        )
        if period is not None:
            for n in [0, -1]:
                ax2.errorbar(
                    df.loc[mask_filter, "hjd"] / period % 1 + n,
                    df.loc[mask_filter, "mag"],
                    df.loc[mask_filter, "magerr"],
                    marker=".",
                    color=colors[band],
                    lw=0
                )

            # invert y axes
            for ax in [ax1, ax2]:
                ax.invert_yaxis()

    ax1.set_xlabel("Time")
    ax1.grid(lw=0.3)
    if period is not None:
        ax2.set_xlabel(f'phase [period={period:4.4g} days]')
        ax2.set_xlim(-1, 1)
        ax2.grid(lw=0.3)

    if save is not None:
        fig.tight_layout()
        plt.savefig(save)


def get_gaiadata(coords, filename):
    """ query vizier to get gaia MG,BPRP values"""
    print(filename)
    # data = Xmatch_Gaia(coords[:, 0], coords[:, 1], max_distance=3.)
    data = np.zeros((len(coords), 9))

    m = data[:, 6]
    M = m + 5 * np.log10(data[:, 0] * 0.001) + 5
    # Mu = m + 5 * np.log10((data[:, 0] - data[:, 1]) * 0.001) + 5
    Ml = m + 5 * np.log10((data[:, 0] + data[:, 1]) * 0.001) + 5

    # Merr = [M-Mu,M-Ml]

    BPRP = data[:, 7] - data[:, 8]

    np.savetxt(filename, np.c_[BPRP, M, M - Ml])


def plot_gaiaHR(figname, data):
    """ Plot the Gaia HR diagram with a sample of objects over-plotted

    source: https://vlas.dev/post/gaia-dr2-hrd/

    """
    # plot the H-R diagram for 1 M stars within 200 pc from the Sun
    plt.rc('text', usetex=True)

    # load background histogram
    h = np.loadtxt('gaia_h.dat')
    xedges = np.loadtxt('gaia_xedges.dat')
    yedges = np.loadtxt('gaia_yedges.dat')

    # make figure
    fig, ax = plt.subplots(figsize=(6, 6))
    cmin = 20
    cmax = None
    if cmin is not None:
        h[h < cmin] = None
    if cmax is not None:
        h[h > cmax] = None

    # f = ax.pcolormesh(xedges, yedges, h.T)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])

    # fill the rest with scatter (set rasterized=True if saving as vector graphics)
    # ax.scatter(bp_rp, mg, alpha=0.05, s=1, color='k', zorder=0)
    ax.invert_yaxis()
    # cb = fig.colorbar(f, ax=ax, pad=0.02)
    ax.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax.set_ylabel(r'$M_G$')
    # cb.set_label(r"$\mathrm{Stellar~density}$")

    # plot sample data
    if np.shape(data)[1] == 2:
        plt.errorbar(data[:, 0], data[:, 1],
                     fmt='r.', lw=0.5)
    if np.shape(data)[1] == 3:
        plt.errorbar(data[:, 0], data[:, 1], data[:, 2],
                     fmt='r.', lw=0.5)
    if np.shape(data)[1] == 4:
        plt.errorbar(data[:, 0], data[:, 1], [data[:, 2] - data[:, 1], data[:, 1] - data[:, 3]],
                     fmt='r.', lw=0.5)

    plt.savefig(figname)

    plt.show()
