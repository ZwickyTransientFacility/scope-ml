#!/usr/bin/env python
"""
Tests for scope.rubin module.

Unit tests (no network required) test flux conversion, band mapping,
and format conversion. Integration tests require a RUBIN_TAP_TOKEN
environment variable.
"""

import os
import numpy as np
import pytest

from scope.rubin import (
    _flux_to_mag,
    _format_as_kowalski,
    DEFAULT_BAND_MAP,
    NANOJANSKY_ZP,
)


class TestFluxToMag:
    """Unit tests for nanoJansky flux to AB magnitude conversion."""

    def test_known_conversion(self):
        """1 nJy should give mag = 31.4 (the AB zero point for nJy)."""
        mag, magerr = _flux_to_mag(1.0, 0.1)
        assert np.isclose(mag, NANOJANSKY_ZP, atol=1e-10)

    def test_known_bright_source(self):
        """10^6 nJy ≈ 1 mJy → mag ≈ 16.4"""
        mag, magerr = _flux_to_mag(1e6, 1e4)
        expected = -2.5 * np.log10(1e6) + NANOJANSKY_ZP  # = 16.4
        assert np.isclose(mag, expected, atol=1e-10)

    def test_error_propagation(self):
        """Check that magerr = (2.5/ln10) * (flux_err/flux)."""
        flux, flux_err = 1000.0, 50.0
        mag, magerr = _flux_to_mag(flux, flux_err)
        expected_magerr = (2.5 / np.log(10)) * (flux_err / flux)
        assert np.isclose(magerr, expected_magerr, atol=1e-10)

    def test_array_input(self):
        """Should handle numpy arrays."""
        fluxes = np.array([100.0, 1000.0, 10000.0])
        errors = np.array([10.0, 100.0, 1000.0])
        mags, magerrs = _flux_to_mag(fluxes, errors)
        assert mags.shape == (3,)
        assert magerrs.shape == (3,)
        # Brighter flux = smaller magnitude
        assert mags[0] > mags[1] > mags[2]

    def test_single_float(self):
        """Should handle single float input."""
        mag, magerr = _flux_to_mag(500.0, 25.0)
        assert isinstance(float(mag), float)
        assert isinstance(float(magerr), float)

    def test_symmetric_error_ratio(self):
        """All sources with 10% flux error should have same magerr."""
        fluxes = np.array([100.0, 1000.0, 10000.0])
        errors = fluxes * 0.1
        _, magerrs = _flux_to_mag(fluxes, errors)
        # All should have the same magerr since err/flux = 0.1 for all
        assert np.allclose(magerrs, magerrs[0], atol=1e-10)


class TestBandMap:
    """Unit tests for band name to filter ID mapping."""

    def test_default_band_map_has_all_bands(self):
        """Default band map should have u, g, r, i, z, y."""
        expected = {"u", "g", "r", "i", "z", "y"}
        assert set(DEFAULT_BAND_MAP.keys()) == expected

    def test_ztf_compatible_bands(self):
        """g=1, r=2, i=3 should match ZTF filter IDs."""
        assert DEFAULT_BAND_MAP["g"] == 1
        assert DEFAULT_BAND_MAP["r"] == 2
        assert DEFAULT_BAND_MAP["i"] == 3

    def test_rubin_only_bands(self):
        """u=0, z=4, y=5 are Rubin-only bands."""
        assert DEFAULT_BAND_MAP["u"] == 0
        assert DEFAULT_BAND_MAP["z"] == 4
        assert DEFAULT_BAND_MAP["y"] == 5


class TestFormatAsKowalski:
    """Unit tests for TAP result → Kowalski format conversion."""

    def _make_rows(self, n=5, objectId=12345, band="g"):
        """Helper to create mock TAP result rows."""
        rows = []
        for i in range(n):
            rows.append(
                {
                    "objectId": objectId,
                    "band": band,
                    "psfFlux": 1000.0 + i * 10,
                    "psfFluxErr": 50.0 + i,
                    "expMidptMJD": 60000.0 + i,
                    "pixelFlags": 0,
                }
            )
        return rows

    def test_basic_conversion(self):
        """Should produce a list of dicts with _id, filter, data keys."""
        rows = self._make_rows()
        result = _format_as_kowalski(rows)
        assert len(result) == 1
        lc = result[0]
        assert "_id" in lc
        assert "filter" in lc
        assert "data" in lc
        assert lc["_id"] == 12345
        assert lc["filter"] == 1  # g band

    def test_data_point_keys(self):
        """Each data point should have hjd, mag, magerr, catflags."""
        rows = self._make_rows(n=3)
        result = _format_as_kowalski(rows)
        for point in result[0]["data"]:
            assert "hjd" in point
            assert "mag" in point
            assert "magerr" in point
            assert "catflags" in point

    def test_mjd_passthrough(self):
        """MJD values should be stored in the 'hjd' field."""
        rows = [
            {
                "objectId": 1,
                "band": "r",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60123.456,
                "pixelFlags": 0,
            }
        ]
        result = _format_as_kowalski(rows)
        assert np.isclose(result[0]["data"][0]["hjd"], 60123.456)

    def test_pixel_flags_to_catflags(self):
        """pixelFlags should be mapped to catflags."""
        rows = [
            {
                "objectId": 1,
                "band": "r",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60000.0,
                "pixelFlags": 42,
            }
        ]
        result = _format_as_kowalski(rows)
        assert result[0]["data"][0]["catflags"] == 42

    def test_multiple_bands_grouped(self):
        """Rows with different bands should produce separate entries."""
        rows_g = self._make_rows(n=3, objectId=1, band="g")
        rows_r = self._make_rows(n=4, objectId=1, band="r")
        result = _format_as_kowalski(rows_g + rows_r)
        assert len(result) == 2
        filters = {lc["filter"] for lc in result}
        assert filters == {1, 2}  # g=1, r=2

    def test_multiple_objects(self):
        """Rows with different objectIds should produce separate entries."""
        rows1 = self._make_rows(n=3, objectId=100, band="g")
        rows2 = self._make_rows(n=3, objectId=200, band="g")
        result = _format_as_kowalski(rows1 + rows2)
        assert len(result) == 2
        ids = {lc["_id"] for lc in result}
        assert ids == {100, 200}

    def test_negative_flux_filtered(self):
        """Rows with non-positive flux should be excluded."""
        rows = [
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": -100.0,
                "psfFluxErr": 10.0,
                "expMidptMJD": 60000.0,
                "pixelFlags": 0,
            },
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 0.0,
                "psfFluxErr": 10.0,
                "expMidptMJD": 60001.0,
                "pixelFlags": 0,
            },
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60002.0,
                "pixelFlags": 0,
            },
        ]
        result = _format_as_kowalski(rows)
        # Only the last row with positive flux should remain
        assert len(result) == 1
        assert len(result[0]["data"]) == 1

    def test_empty_input(self):
        """Empty input should return empty list."""
        result = _format_as_kowalski([])
        assert result == []

    def test_custom_band_map(self):
        """Should use a custom band map if provided."""
        custom_map = {"g": 10, "r": 20}
        rows = self._make_rows(n=2, band="g")
        result = _format_as_kowalski(rows, band_map=custom_map)
        assert result[0]["filter"] == 10

    def test_unknown_band_returns_negative_one(self):
        """Unknown band should map to filter ID -1."""
        rows = self._make_rows(n=2, band="x")
        result = _format_as_kowalski(rows)
        assert result[0]["filter"] == -1

    def test_magnitude_values_reasonable(self):
        """Converted magnitudes should be in a reasonable range."""
        rows = self._make_rows(n=10)
        result = _format_as_kowalski(rows)
        for point in result[0]["data"]:
            # Typical astronomical magnitudes are 0-30
            assert 0 < point["mag"] < 40
            assert point["magerr"] > 0


class TestMJDHandling:
    """Tests for MJD timestamp handling."""

    def test_mjd_preserved_exactly(self):
        """MJD should be stored exactly as-is (no HJD conversion)."""
        mjd = 60457.123456789
        rows = [
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": mjd,
                "pixelFlags": 0,
            }
        ]
        result = _format_as_kowalski(rows)
        assert result[0]["data"][0]["hjd"] == mjd

    def test_time_ordering_not_enforced(self):
        """Format conversion should not reorder points (that's sort_lightcurve's job)."""
        rows = [
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60002.0,
                "pixelFlags": 0,
            },
            {
                "objectId": 1,
                "band": "g",
                "psfFlux": 1000.0,
                "psfFluxErr": 50.0,
                "expMidptMJD": 60001.0,
                "pixelFlags": 0,
            },
        ]
        result = _format_as_kowalski(rows)
        times = [p["hjd"] for p in result[0]["data"]]
        # Should preserve original order
        assert times[0] == 60002.0
        assert times[1] == 60001.0


# --- Integration tests (require RUBIN_TAP_TOKEN) ---

# DP1 footprint: Extended Chandra Deep Field South (ECDFS)
# Center ~(53.1, -28.1), ~9.6 sq deg
DP1_TEST_RA = 53.1
DP1_TEST_DEC = -28.1


@pytest.mark.skipif(
    os.environ.get("RUBIN_TAP_TOKEN") is None,
    reason="Integration test requires RUBIN_TAP_TOKEN environment variable",
)
class TestIntegrationRubinTAP:
    """Integration tests that connect to the real Rubin TAP service."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from scope.rubin import RubinTAPClient

        self.client = RubinTAPClient(
            tap_url="https://data.lsst.cloud/api/tap",
            token=os.environ["RUBIN_TAP_TOKEN"],
        )

    def test_cone_search(self):
        """Cone search should return objects as a dict."""
        try:
            objects = self.client.get_objects_by_cone(
                ra=DP1_TEST_RA,
                dec=DP1_TEST_DEC,
                radius_arcsec=30.0,
                limit=10,
            )
        except Exception as e:
            pytest.skip(f"TAP service unavailable: {e}")
        assert isinstance(objects, dict)
        if len(objects) > 0:
            first = next(iter(objects.values()))
            assert "coord_ra" in first
            assert "coord_dec" in first

    def test_lightcurve_retrieval(self):
        """Should retrieve lightcurves for discovered objects."""
        try:
            objects = self.client.get_objects_by_cone(
                ra=DP1_TEST_RA,
                dec=DP1_TEST_DEC,
                radius_arcsec=30.0,
                limit=5,
            )
        except Exception as e:
            pytest.skip(f"TAP service unavailable: {e}")
        if len(objects) == 0:
            pytest.skip("No objects found in cone search")

        objectids = list(objects.keys())[:3]
        lcs = self.client.get_lightcurves(objectids, bands=["g", "r"])
        assert isinstance(lcs, list)
        if len(lcs) > 0:
            lc = lcs[0]
            assert "_id" in lc
            assert "filter" in lc
            assert "data" in lc
            assert len(lc["data"]) > 0
            point = lc["data"][0]
            assert "hjd" in point
            assert "mag" in point
            assert "magerr" in point
            assert "catflags" in point

    def test_pipeline_compatibility(self):
        """Lightcurve format should be compatible with sort_lightcurve."""
        from scope.utils import sort_lightcurve

        try:
            objects = self.client.get_objects_by_cone(
                ra=DP1_TEST_RA,
                dec=DP1_TEST_DEC,
                radius_arcsec=30.0,
                limit=5,
            )
        except Exception as e:
            pytest.skip(f"TAP service unavailable: {e}")
        if len(objects) == 0:
            pytest.skip("No objects found in cone search")

        objectids = list(objects.keys())[:1]
        lcs = self.client.get_lightcurves(objectids)
        if len(lcs) == 0:
            pytest.skip("No lightcurves retrieved")

        lc = lcs[0]
        tme = np.array([[p["hjd"], p["mag"], p["magerr"]] for p in lc["data"]])
        t, m, e = tme.T
        t_sorted, m_sorted, e_sorted = sort_lightcurve(t, m, e)

        # Should be monotonically increasing
        assert np.all(np.diff(t_sorted) >= 0)
        assert len(t_sorted) == len(t)
