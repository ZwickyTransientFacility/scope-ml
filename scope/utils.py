__all__ = [
    "load_config",
]

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(config_path: str):
    """
    Load config and secrets
    """
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    return config


def get_lc(ra, dec, kow, outputname=None, radius=1.5, remove_flagged=True):
    """get a lightcurve from Kowalski """

    q = {
        "query_type": "cone_search",
        "query": {
            "object_coordinates": {
                "cone_search_radius": 2,
                "cone_search_unit": "arcsec",
                "radec": {
                    "target": [ra, dec]
                }
            },
            "catalogs": {
                "DATABASE": {
                    "filter": {},
                    "projection": {
                        "_id": 1,
                        "data.hjd": 1,
                        "data.fid": 1,
                        "data.filter": 1,
                        "data.mag": 1,
                        "data.magerr": 1,
                        "data.ra": 1,
                        "data.dec": 1,
                        "data.programid": 1,
                        "data.catflags": 1}
                }
            }
        },
        "kwargs": {
            "filter_first": False
        }
    }

    #
    ra_0 = ra
    dec_0 = dec

    # run query
    r = kow.query(q)
    data = r.get('data')

    #
    key = list(data.keys())[0]
    data = data[key]
    key = list(data.keys())[0]
    data = data[key]

    # storage for outputdata
    hjd, mag, magerr = [], [], []
    ra, dec, fid, pid, catflags = [], [], [], [], []

    # loop over ids to get the data
    for datlist in data:
        objid = str(datlist["_id"])
        _fid = int(str(objid)[7])

        dat = datlist["data"]

        for dic in dat:
            hjd.append(dic["hjd"])
            mag.append(dic["mag"])
            magerr.append(dic["magerr"])
            ra.append(dic["ra"])
            dec.append(dic["dec"])
            fid.append(_fid)
            pid.append(dic["programid"])
            catflags.append(dic["catflags"])

    # combine into one array
    lightcurve = np.c_[hjd, mag, magerr, fid, pid, ra, dec, np.zeros_like(hjd), catflags]

    if outputname is None:
        outputname = 'lc_%0.4f_%0.4f.dat' % (ra_0, dec_0)

    np.savetxt(outputname, lightcurve, fmt='%14.14g')
    # return lightcurve


def make_lcfig(lcfile, p=None, title=None, savename=None, splitbands=False):
    """ given a lightcurve files (JD,mag,mag_e,filter)
    lcfile : a file with the lightcurve (JD,mag,mag_e,filter)
    p : float, the period to fold the lightcurve. If not specified, only the regular lighcurve is shown
    title: str, the title of the figure
    savename: str, the name used to save the figure
    splitbands: bool, split the g,r,i bands
    """

    # the start of ZTF
    ZTF_JD = 2458178.50000  # 2019 march 1

    # plot the figure
    colors = {1: '#28a745',
              2: '#dc3545',
              3: '#343a40',
              'default': '#00415a', }

    markers = {1: '^',
               2: '>',
               3: '<'}

    # load the light curve
    lc = np.loadtxt(lcfile)

    # remove bad data
    lc = lc[lc[:, 8] == 0]

    # make a figure
    fig = plt.figure(constrained_layout=False, figsize=(16, 9))
    fig.suptitle(title, fontsize=32)

    if p is None:
        ax1 = fig.add_subplot(111)
    if p is not None:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

    # plot the different ZTF bands
    for f in [1, 2, 3]:
        m = lc[:, 3] == f
        ax1.errorbar(lc[m, 0] - ZTF_JD, lc[m, 1], lc[m, 2],
                     marker='.', color=colors[f], lw=0)
        if p is not None:
            for n in [0, -1]:
                ax2.errorbar(lc[m, 0] / p % 1 + n, lc[m, 1], lc[m, 2],
                             marker=markers[f], color=colors[f], lw=0)

                # invert axes
    for ax in [ax1, ax2]:
        ax.invert_yaxis()

    # set axes etc
    ax1.set_xlabel('Time')
    if p is not None:
        ax2.set_xlabel('phase [p=%4.4g]' % p)
        ax2.set_xlim(-1, 1)

    if savename is not None:
        plt.savefig(savename)
    # plt.show()
    plt.close()


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
