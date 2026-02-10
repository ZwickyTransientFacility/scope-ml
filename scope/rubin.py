#!/usr/bin/env python
"""
Rubin LSST TAP client for scope-ml.

Provides access to Rubin Data Preview (DP1) data via the TAP API,
converting forced photometry lightcurves into the same format consumed
by the scope-ml feature generation pipeline.
"""

import numpy as np
import warnings

try:
    import pyvo
    import pyvo.auth

    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

from .utils import parse_load_config

# Band name -> integer filter ID mapping
# g=1, r=2, i=3 match ZTF filter IDs; u, z, y are Rubin-only
DEFAULT_BAND_MAP = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

# AB magnitude zero point for nanoJansky: mag = -2.5 * log10(flux_nJy) + 31.4
NANOJANSKY_ZP = 31.4


def _flux_to_mag(flux_nJy, flux_err_nJy):
    """
    Convert flux in nanoJansky to AB magnitude and magnitude error.

    Parameters
    ----------
    flux_nJy : float or array-like
        Flux in nanoJansky. Must be > 0.
    flux_err_nJy : float or array-like
        Flux uncertainty in nanoJansky. Must be > 0.

    Returns
    -------
    mag : float or ndarray
        AB magnitude
    magerr : float or ndarray
        Magnitude uncertainty
    """
    flux_nJy = np.asarray(flux_nJy, dtype=float)
    flux_err_nJy = np.asarray(flux_err_nJy, dtype=float)

    mag = -2.5 * np.log10(flux_nJy) + NANOJANSKY_ZP
    magerr = (2.5 / np.log(10)) * (flux_err_nJy / flux_nJy)

    return mag, magerr


def _format_as_kowalski(rows, band_map=None):
    """
    Convert TAP query result rows into Kowalski-format lightcurve dicts.

    The output matches the format returned by ``get_lightcurves_via_ids``
    in ``scope.fritz``:
        [{'_id': objectId, 'filter': band_int, 'data': [
            {'hjd': mjd, 'mag': m, 'magerr': e, 'catflags': flags}, ...
        ]}, ...]

    One entry per (objectId, band) combination is returned.

    Parameters
    ----------
    rows : list of dict
        Each dict must have keys: objectId, band, psfFlux, psfFluxErr,
        expMidptMJD, pixelFlags.
    band_map : dict, optional
        Mapping from band name (str) to integer filter ID.

    Returns
    -------
    list of dict
        Kowalski-format lightcurve dicts.
    """
    if band_map is None:
        band_map = DEFAULT_BAND_MAP

    # Group by (objectId, band)
    grouped = {}
    for row in rows:
        key = (int(row["objectId"]), str(row["band"]))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    result = []
    for (obj_id, band), points in grouped.items():
        filt = band_map.get(band, -1)
        data = []
        for p in points:
            flux = float(p["psfFlux"])
            flux_err = float(p["psfFluxErr"])
            if flux <= 0 or flux_err <= 0:
                continue
            mag, magerr = _flux_to_mag(flux, flux_err)
            data.append(
                {
                    "hjd": float(p["expMidptMJD"]),
                    "mag": float(mag),
                    "magerr": float(magerr),
                    "catflags": int(p["pixelFlags"]),
                }
            )
        if len(data) > 0:
            result.append({"_id": obj_id, "filter": filt, "data": data})

    return result


class RubinTAPClient:
    """
    Client for querying Rubin LSST data via the TAP API.

    Parameters
    ----------
    tap_url : str
        URL of the TAP service endpoint.
    token : str
        Authentication token for the TAP service.
    timeout : int, optional
        Query timeout in seconds (default 300).
    band_map : dict, optional
        Mapping from band name to integer filter ID.
    config : dict, optional
        Rubin config section. If provided, overrides tap_url/token/timeout
        unless those are explicitly given.
    """

    def __init__(
        self,
        tap_url=None,
        token=None,
        timeout=None,
        band_map=None,
        config=None,
    ):
        if not HAS_PYVO:
            raise ImportError(
                "pyvo is required for Rubin TAP access. "
                "Install it with: pip install pyvo"
            )

        # Resolve from config if provided
        if config is not None:
            tap_url = tap_url or config.get("tap_url", "https://data.lsst.cloud/api/tap")
            token = token or config.get("token")
            timeout = timeout if timeout is not None else config.get("timeout", 300)
            band_map = band_map or config.get("band_map", DEFAULT_BAND_MAP)

        if token is None:
            raise ValueError(
                "Rubin TAP token is required. Set it in config.yaml under "
                "rubin.token or via the RUBIN_TAP_TOKEN environment variable."
            )

        self.tap_url = tap_url
        self.token = token
        self.timeout = timeout or 300
        self.band_map = band_map or DEFAULT_BAND_MAP

        # Tables (configurable via config)
        if config is not None and "tables" in config:
            tables = config["tables"]
            self.table_object = tables.get("object", "dp1.Object")
            self.table_forced_source = tables.get("forced_source", "dp1.ForcedSource")
            self.table_visit = tables.get("visit", "dp1.Visit")
        else:
            self.table_object = "dp1.Object"
            self.table_forced_source = "dp1.ForcedSource"
            self.table_visit = "dp1.Visit"

        # Set up authenticated TAP service via Bearer token
        import requests

        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {self.token}"
        self.service = pyvo.dal.TAPService(self.tap_url, session=session)

    def get_objects_by_cone(self, ra, dec, radius_arcsec, limit=10000):
        """
        Find objects within a cone search region.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        limit : int, optional
            Maximum number of objects to return (default 10000).

        Returns
        -------
        dict
            {objectId: {'coord_ra': ra_deg, 'coord_dec': dec_deg}, ...}
        """
        radius_deg = radius_arcsec / 3600.0

        top_clause = f"TOP {limit}" if limit is not None else ""
        query = f"""
        SELECT {top_clause} objectId, coord_ra, coord_dec
        FROM {self.table_object}
        WHERE CONTAINS(
            POINT('ICRS', coord_ra, coord_dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
        AND detect_isIsolated = 1
        """

        result = self.service.search(query)
        objects = {}
        for row in result:
            oid = int(row["objectId"])
            objects[oid] = {
                "coord_ra": float(row["coord_ra"]),
                "coord_dec": float(row["coord_dec"]),
            }
        return objects

    def get_lightcurves(self, objectids, bands=None, batch_size=1000):
        """
        Retrieve forced photometry lightcurves for a list of object IDs.

        Parameters
        ----------
        objectids : list of int
            Object IDs to query.
        bands : list of str, optional
            Band names to filter (e.g., ['g', 'r']). If None, all bands.
        batch_size : int, optional
            Number of IDs per query batch (default 1000).

        Returns
        -------
        list of dict
            Kowalski-format lightcurve dicts.
        """
        if len(objectids) == 0:
            return []

        all_rows = []

        # Batch the queries
        for i in range(0, len(objectids), batch_size):
            batch = objectids[i : i + batch_size]
            id_list = ", ".join(str(oid) for oid in batch)

            band_clause = ""
            if bands is not None and len(bands) > 0:
                band_values = ", ".join(f"'{b}'" for b in bands)
                band_clause = f"AND fs.band IN ({band_values})"

            query = f"""
            SELECT fs.objectId, fs.band, fs.psfFlux, fs.psfFluxErr,
                   v.expMidptMJD,
                   (fs.pixelFlags_bad + fs.pixelFlags_cr
                    + fs.pixelFlags_edge + fs.pixelFlags_saturated
                    + fs.pixelFlags_suspect) AS pixelFlags
            FROM {self.table_forced_source} AS fs
            JOIN {self.table_visit} AS v
                ON fs.visit = v.visit
            WHERE fs.objectId IN ({id_list})
            AND fs.psfFlux > 0
            AND fs.psfFluxErr > 0
            {band_clause}
            """

            try:
                # Use async TAP for large queries
                if len(batch) > 100:
                    job = self.service.submit_job(query)
                    job.run()
                    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=self.timeout)
                    if job.phase == "COMPLETED":
                        result = job.fetch_result()
                    else:
                        warnings.warn(
                            f"TAP async job failed with phase: {job.phase}. "
                            f"Skipping batch starting at index {i}."
                        )
                        continue
                else:
                    result = self.service.search(query)

                for row in result:
                    all_rows.append(
                        {
                            "objectId": row["objectId"],
                            "band": row["band"],
                            "psfFlux": row["psfFlux"],
                            "psfFluxErr": row["psfFluxErr"],
                            "expMidptMJD": row["expMidptMJD"],
                            "pixelFlags": row["pixelFlags"],
                        }
                    )

            except Exception as e:
                warnings.warn(
                    f"TAP query failed for batch starting at index {i}: {e}"
                )
                continue

        return _format_as_kowalski(all_rows, band_map=self.band_map)

    def get_lightcurves_for_cone(
        self, ra, dec, radius_arcsec, bands=None, limit=10000
    ):
        """
        Convenience method: cone search + lightcurve retrieval in one call.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcsec : float
            Search radius in arcseconds.
        bands : list of str, optional
            Band names to filter.
        limit : int, optional
            Maximum number of objects from cone search.

        Returns
        -------
        objects : dict
            Object metadata from cone search.
        lightcurves : list of dict
            Kowalski-format lightcurve dicts.
        """
        objects = self.get_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
        if len(objects) == 0:
            return objects, []

        objectids = list(objects.keys())
        lightcurves = self.get_lightcurves(objectids, bands=bands)
        return objects, lightcurves
