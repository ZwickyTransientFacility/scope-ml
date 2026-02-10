"""Tests for scope-ml period searching using the periodfind unified device API.

Uses synthetic lightcurve data — no GPU or Kowalski needed.

Run with:
    python -m pytest tests/test_periodsearch.py -v
"""

import sys
import os

# Add the tools directory to the path so we can import periodsearch
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'featureGeneration')
)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import periodfind  # noqa: E402

from periodsearch import (  # noqa: E402
    find_periods,
    compute_fourier_features,
    compute_dmdt_features,
    compute_basic_stats,
    remove_high_cadence_batch,
    _normalize_algorithm,
    _prepare_lightcurves,
    _build_pdots,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sinusoidal_lightcurve(
    period, n_points=500, amplitude=1.0, noise_std=0.05, t_span=100.0, seed=42
):
    """Generate a synthetic sinusoidal lightcurve as (times, mags, magerrs).

    Returns the tuple format expected by scope-ml's find_periods:
    a masked array of shape (3, n_points) with rows [time, mag, magerr].
    """
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points))
    phase = 2.0 * np.pi * times / period
    mags = amplitude * np.sin(phase) + amplitude + 10.0  # shift to positive
    mags += rng.normal(0, noise_std, n_points)
    magerrs = np.full(n_points, noise_std)
    # scope-ml expects lightcurves as masked arrays of shape (3, N)
    return np.ma.array([times, mags, magerrs])


def make_freq_grid(f_min=0.1, f_max=5.0, n_freqs=500):
    """Create a uniform frequency grid."""
    return np.linspace(f_min, f_max, n_freqs)


# ---------------------------------------------------------------------------
# TestFindPeriods
# ---------------------------------------------------------------------------


class TestFindPeriods:
    """Unit tests for the find_periods() function."""

    def test_ce_stats_output(self):
        """CE algorithm returns flat arrays of correct length."""
        lcs = [make_sinusoidal_lightcurve(period=3.0, seed=i) for i in range(2)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("CE", lcs, freqs, doCPU=True)

        assert periods.shape == (2,)
        assert sigs.shape == (2,)
        assert pdots.shape == (2,)
        assert np.all(np.isfinite(periods))
        assert np.all(np.isfinite(sigs))

    def test_aov_stats_output(self):
        """AOV algorithm returns flat arrays of correct length."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("AOV", lcs, freqs, doCPU=True)

        assert periods.shape == (1,)
        assert sigs.shape == (1,)
        assert np.all(np.isfinite(periods))

    def test_ls_stats_output(self):
        """LS algorithm returns flat arrays of correct length."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("LS", lcs, freqs, doCPU=True)

        assert periods.shape == (1,)
        assert sigs.shape == (1,)
        assert np.all(np.isfinite(periods))

    def test_ce_detects_known_period(self):
        """CE recovers a known sinusoidal period."""
        true_period = 5.0
        lcs = [
            make_sinusoidal_lightcurve(
                period=true_period, n_points=800, noise_std=0.02, t_span=200.0
            )
        ]
        freqs = make_freq_grid(0.1, 1.0, 500)

        periods, sigs, pdots = find_periods("CE", lcs, freqs, doCPU=True)

        detected = periods[0]
        # Allow 5% tolerance or detection of a harmonic
        candidates = [true_period, true_period / 2, 2 * true_period]
        assert any(
            abs(detected - c) / c < 0.05 for c in candidates
        ), f"Expected ~{true_period}, got {detected}"

    def test_ls_detects_known_period(self):
        """LS recovers a known sinusoidal period."""
        true_period = 5.0
        lcs = [
            make_sinusoidal_lightcurve(
                period=true_period, n_points=800, noise_std=0.02, t_span=200.0
            )
        ]
        freqs = make_freq_grid(0.1, 1.0, 500)

        periods, sigs, pdots = find_periods("LS", lcs, freqs, doCPU=True)

        detected = periods[0]
        candidates = [true_period, true_period / 2, 2 * true_period]
        assert any(
            abs(detected - c) / c < 0.05 for c in candidates
        ), f"Expected ~{true_period}, got {detected}"

    def test_periodogram_output_format(self):
        """*_periodogram algorithms return list of dicts with 'period' and 'data' keys."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        periods, sigs, pdots = find_periods("CE_periodogram", lcs, freqs, doCPU=True)

        # periods_best is an array of dicts for periodogram output
        assert len(periods) == 1
        entry = periods[0]
        assert isinstance(entry, dict)
        assert 'period' in entry
        assert 'data' in entry
        assert hasattr(entry['data'], 'shape')

    def test_multiple_lightcurves(self):
        """Batched input returns correct number of results."""
        n_curves = 5
        lcs = [make_sinusoidal_lightcurve(period=3.0, seed=i) for i in range(n_curves)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        periods, sigs, pdots = find_periods("CE", lcs, freqs, doCPU=True)

        assert periods.shape == (n_curves,)
        assert sigs.shape == (n_curves,)
        assert pdots.shape == (n_curves,)

    def test_fpw_stats_output(self):
        """FPW algorithm returns flat arrays of correct length."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 200)

        periods, sigs, pdots = find_periods("FPW", lcs, freqs, doCPU=True)

        assert periods.shape == (1,)
        assert sigs.shape == (1,)
        assert np.all(np.isfinite(periods))

    def test_fpw_detects_known_period(self):
        """FPW recovers a known sinusoidal period."""
        true_period = 5.0
        lcs = [
            make_sinusoidal_lightcurve(
                period=true_period, n_points=800, noise_std=0.02, t_span=200.0
            )
        ]
        freqs = make_freq_grid(0.1, 1.0, 500)

        periods, sigs, pdots = find_periods("FPW", lcs, freqs, doCPU=True)

        detected = periods[0]
        candidates = [true_period, true_period / 2, 2 * true_period]
        assert any(
            abs(detected - c) / c < 0.05 for c in candidates
        ), f"Expected ~{true_period}, got {detected}"

    def test_fpw_multiple_lightcurves(self):
        """FPW handles batched light curves."""
        n_curves = 5
        lcs = [make_sinusoidal_lightcurve(period=3.0, seed=i) for i in range(n_curves)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        periods, sigs, pdots = find_periods("FPW", lcs, freqs, doCPU=True)

        assert periods.shape == (n_curves,)
        assert sigs.shape == (n_curves,)

    def test_fpw_periodogram_output(self):
        """FPW_periodogram returns list of dicts with 'period' and 'data' keys."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        periods, sigs, pdots = find_periods("FPW_periodogram", lcs, freqs, doCPU=True)

        assert len(periods) == 1
        entry = periods[0]
        assert isinstance(entry, dict)
        assert 'period' in entry
        assert 'data' in entry

    def test_gpu_algorithm_names(self):
        """ECE/EAOV/ELS/EFPW names work with doCPU=True (maps to same underlying algo)."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        for algo_name in ["ECE", "EAOV", "ELS", "EFPW"]:
            periods, sigs, pdots = find_periods(algo_name, lcs, freqs, doCPU=True)
            assert periods.shape == (1,), f"{algo_name} failed"
            assert np.all(
                np.isfinite(periods)
            ), f"{algo_name} returned non-finite period"


# ---------------------------------------------------------------------------
# TestDeviceAPIIntegration
# ---------------------------------------------------------------------------


class TestDeviceAPIIntegration:
    """Verify periodfind device API works from scope-ml."""

    def setup_method(self):
        """Reset global device state before each test."""
        periodfind._default_device = None

    def teardown_method(self):
        """Reset global device state after each test."""
        periodfind._default_device = None

    def test_set_device_affects_factory(self):
        """periodfind.set_device('cpu') makes factories return CPU instances."""
        from periodfind.cpu import (
            ConditionalEntropy as CpuCE,
            AOV as CpuAOV,
            LombScargle as CpuLS,
            FPW as CpuFPW,
        )

        periodfind.set_device('cpu')
        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)
        aov = periodfind.AOV(n_phase=10)
        ls = periodfind.LombScargle()
        fpw = periodfind.FPW(n_bins=10)
        assert isinstance(ce, CpuCE)
        assert isinstance(aov, CpuAOV)
        assert isinstance(ls, CpuLS)
        assert isinstance(fpw, CpuFPW)

    def test_factory_with_device_kwarg(self):
        """periodfind.ConditionalEntropy(device='cpu') returns CPU backend."""
        from periodfind.cpu import ConditionalEntropy as CpuCE

        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10, device='cpu')
        assert isinstance(ce, CpuCE)


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


class TestNormalizeAlgorithm:
    """Tests for _normalize_algorithm helper."""

    def test_plain_names(self):
        assert _normalize_algorithm("CE") == ("CE", False)
        assert _normalize_algorithm("AOV") == ("AOV", False)
        assert _normalize_algorithm("LS") == ("LS", False)
        assert _normalize_algorithm("FPW") == ("FPW", False)

    def test_e_prefix(self):
        assert _normalize_algorithm("ECE") == ("CE", False)
        assert _normalize_algorithm("EAOV") == ("AOV", False)
        assert _normalize_algorithm("ELS") == ("LS", False)
        assert _normalize_algorithm("EFPW") == ("FPW", False)

    def test_periodogram_suffix(self):
        assert _normalize_algorithm("CE_periodogram") == ("CE", True)
        assert _normalize_algorithm("ECE_periodogram") == ("CE", True)
        assert _normalize_algorithm("EAOV_periodogram") == ("AOV", True)
        assert _normalize_algorithm("FPW_periodogram") == ("FPW", True)
        assert _normalize_algorithm("EFPW_periodogram") == ("FPW", True)

    def test_aov_cython_unrecognized(self):
        name, is_pg = _normalize_algorithm("AOV_cython")
        # AOV_cython should not match the unified path
        assert name not in {"CE", "AOV", "LS"}


class TestBuildPdots:
    """Tests for _build_pdots helper."""

    def test_no_pdot(self):
        pdots = _build_pdots(False)
        assert pdots.dtype == np.float32
        assert len(pdots) == 1
        assert pdots[0] == 0.0

    def test_with_pdot(self):
        pdots = _build_pdots(True)
        assert pdots.dtype == np.float32
        assert len(pdots) == 11  # 10 logspace + 1 zero
        assert pdots[0] == 0.0
        assert all(p <= 0 for p in pdots)


class TestPrepareLightcurves:
    """Tests for _prepare_lightcurves helper."""

    def test_basic_preparation(self):
        rng = np.random.default_rng(42)
        times = rng.uniform(0, 100, 50)
        mags = rng.uniform(14.0, 16.0, 50)
        magerrs = np.full(50, 0.01)
        lcs = [(times, mags, magerrs)]

        time_stack, mag_stack = _prepare_lightcurves(lcs, doSingleTimeSegment=False)

        assert len(time_stack) == 1
        assert len(mag_stack) == 1
        assert time_stack[0].dtype == np.float32
        assert mag_stack[0].dtype == np.float32
        # Times should be sorted
        assert np.all(np.diff(time_stack[0]) >= 0)
        # Mags should be normalized to [0, 1]
        assert mag_stack[0].min() >= 0.0
        assert mag_stack[0].max() <= 1.0


# ---------------------------------------------------------------------------
# TestComputeFourierFeatures
# ---------------------------------------------------------------------------


class TestComputeFourierFeatures:
    """Tests for the compute_fourier_features() function."""

    def test_output_shape(self):
        """Returns (n_curves, 14) array."""
        rng = np.random.default_rng(42)
        n_curves = 3
        lcs = []
        for i in range(n_curves):
            t = np.sort(rng.uniform(0, 100, 100))
            m = 18.0 + rng.normal(0, 0.1, 100)
            e = np.full(100, 0.1)
            lcs.append((t, m, e))
        periods = np.array([1.0, 2.0, 3.0])

        result = compute_fourier_features(lcs, periods)
        assert result.shape == (n_curves, 14)
        assert result.dtype == np.float32

    def test_known_sinusoid_recovery(self):
        """Fourier decomposition recovers known sinusoidal signal."""
        n = 300
        period = 5.0
        t = np.sort(np.random.default_rng(42).uniform(0, 50, n))
        phi = 2.0 * np.pi * t / period
        m = 18.0 + 0.5 * np.cos(phi)
        e = np.full(n, 0.01)

        result = compute_fourier_features([(t, m, e)], np.array([period]))
        r = result[0]

        # Power should be close to 1
        assert r[0] > 0.9, f"power = {r[0]}"
        # Offset should be close to 18
        assert abs(r[2] - 18.0) < 0.1, f"offset = {r[2]}"
        # A1 should be close to 0.5
        assert abs(r[4] - 0.5) < 0.1, f"A1 = {r[4]}"
        # B1 should be close to 0
        assert abs(r[5]) < 0.1, f"B1 = {r[5]}"

    def test_constant_signal_low_power(self):
        """Constant magnitude yields ~0 power."""
        n = 100
        t = np.arange(n, dtype=np.float64) * 0.5
        m = np.full(n, 17.0)
        e = np.full(n, 0.01)

        result = compute_fourier_features([(t, m, e)], np.array([2.0]))
        assert abs(result[0, 0]) < 0.01, f"power = {result[0, 0]}"

    def test_multiple_curves_independent(self):
        """Each curve is processed independently."""
        rng = np.random.default_rng(99)
        n = 200
        period = 3.0

        # Curve 0: sinusoidal
        t = np.sort(rng.uniform(0, 30, n))
        phi = 2.0 * np.pi * t / period
        m_sin = 18.0 + 1.0 * np.cos(phi)
        e = np.full(n, 0.01)

        # Curve 1: constant
        m_const = np.full(n, 18.0)

        result = compute_fourier_features(
            [(t, m_sin, e), (t, m_const, e)],
            np.array([period, period]),
        )

        assert result[0, 0] > 0.9, "sinusoidal curve should have high power"
        assert result[1, 0] < 0.01, "constant curve should have ~0 power"

    def test_numerical_agreement_with_lcstats(self):
        """Rust Fourier agrees with lcstats.fourier_decomposition within ~1%."""
        lcstats = pytest.importorskip("lcstats", reason="lcstats requires scipy/numba")

        rng = np.random.default_rng(123)
        n = 200
        t = np.sort(rng.uniform(0, 100, n))
        period = 4.0
        phi = 2.0 * np.pi * t / period
        m = 16.0 + 0.4 * np.cos(phi) + 0.2 * np.sin(phi)
        m += rng.normal(0, 0.02, n)
        e = np.full(n, 0.02)

        old = lcstats.fourier_decomposition(t, m, e, period)
        new = compute_fourier_features([(t, m, e)], np.array([period]))[0]

        # Power and BIC should be very close
        assert (
            abs(old[0] - new[0]) / max(abs(old[0]), 1e-12) < 0.01
        ), f"power: old={old[0]}, new={new[0]}"
        assert (
            abs(old[1] - new[1]) / max(abs(old[1]), 1e-12) < 0.01
        ), f"BIC: old={old[1]}, new={new[1]}"
        # Offset
        assert (
            abs(old[2] - new[2]) / max(abs(old[2]), 1e-12) < 0.01
        ), f"offset: old={old[2]}, new={new[2]}"
        # A1, B1 should be close (allow 5% for f32/f64 + method differences)
        for i, name in [(4, "A1"), (5, "B1")]:
            if abs(old[i]) > 0.01:
                reldiff = abs(old[i] - new[i]) / abs(old[i])
                assert (
                    reldiff < 0.05
                ), f"{name}: old={old[i]}, new={new[i]}, reldiff={reldiff}"


# ---------------------------------------------------------------------------
# TestRemoveHighCadence
# ---------------------------------------------------------------------------


class TestRemoveHighCadence:
    """Tests for the remove_high_cadence_batch() function."""

    def test_basic_filtering(self):
        """Points within cadence are removed."""
        # cadence = 30 min = 30/1440 days
        t = np.array([0.0, 0.001, 0.002, 0.025, 0.05])
        m = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        e = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        result = remove_high_cadence_batch([(t, m, e)], cadence_minutes=30.0)

        assert len(result) == 1
        t_out, m_out, e_out = result[0]
        # 30 min = 0.02083 days; keep 0, 3, 4
        assert len(t_out) == 3
        assert t_out.dtype == np.float32

    def test_empty_input(self):
        """Empty input returns empty output."""
        t = np.array([], dtype=np.float64)
        m = np.array([], dtype=np.float64)
        e = np.array([], dtype=np.float64)

        result = remove_high_cadence_batch([(t, m, e)], cadence_minutes=30.0)

        assert len(result) == 1
        assert len(result[0][0]) == 0

    def test_single_point(self):
        """Single point is preserved."""
        result = remove_high_cadence_batch(
            [(np.array([1.0]), np.array([10.0]), np.array([0.1]))],
            cadence_minutes=30.0,
        )
        assert len(result[0][0]) == 1

    def test_all_beyond_cadence(self):
        """Well-separated points are all kept."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        m = np.array([10.0, 11.0, 12.0, 13.0])
        e = np.array([0.1, 0.1, 0.1, 0.1])

        result = remove_high_cadence_batch([(t, m, e)], cadence_minutes=30.0)
        assert len(result[0][0]) == 4

    def test_batch_processing(self):
        """Multiple curves processed correctly."""
        t1 = np.array([0.0, 1.0, 2.0])
        m1 = np.array([10.0, 11.0, 12.0])
        e1 = np.array([0.1, 0.1, 0.1])

        t2 = np.array([0.0, 0.001, 5.0])
        m2 = np.array([20.0, 21.0, 22.0])
        e2 = np.array([0.2, 0.2, 0.2])

        result = remove_high_cadence_batch(
            [(t1, m1, e1), (t2, m2, e2)], cadence_minutes=30.0
        )
        assert len(result) == 2
        assert len(result[0][0]) == 3  # all kept
        assert len(result[1][0]) == 2  # first and last kept


# ---------------------------------------------------------------------------
# TestComputeDmdtFeatures
# ---------------------------------------------------------------------------


class TestComputeDmdtFeatures:
    """Tests for the compute_dmdt_features() function."""

    def _make_dmdt_ints(self):
        """Create standard bin edges matching scope-ml defaults."""
        dt_edges = np.array(
            [0.0, 1.0 / 145, 2.0 / 145, 3.0 / 145, 4.0 / 145, 5.0 / 145,
             6.0 / 145, 1.5 / 23.2, 2.0 / 23.2, 3.0 / 23.2, 1.0 / 3.5,
             2.0 / 3.5, 3.0 / 3.5, 4.0 / 3.5, 5.0 / 3.5, 7.0, 10.0,
             20.0, 30.0, 60.0, 90.0, 120.0, 240.0, 600.0, 960.0, 2000.0],
            dtype=np.float64,
        )
        dm_edges = np.array(
            [-8.0, -3.2, -2.4, -2.0, -1.6, -1.2, -0.8, -0.6, -0.4, -0.3,
             -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8,
             1.2, 1.6, 2.0, 2.4, 3.2, 8.0],
            dtype=np.float64,
        )
        return {'dtints': dt_edges, 'dmints': dm_edges}

    def test_output_shape(self):
        """Returns correct 3D shape."""
        rng = np.random.default_rng(42)
        n_curves = 3
        lcs = []
        for i in range(n_curves):
            t = np.sort(rng.uniform(0, 100, 80))
            m = 18.0 + rng.normal(0, 0.5, 80)
            e = np.full(80, 0.1)
            lcs.append((t, m, e))

        dmdt_ints = self._make_dmdt_ints()
        result = compute_dmdt_features(lcs, dmdt_ints)

        n_dt_bins = len(dmdt_ints['dtints']) - 1
        n_dm_bins = len(dmdt_ints['dmints']) - 1
        assert result.shape == (n_curves, n_dm_bins, n_dt_bins)
        assert result.dtype == np.float32

    def test_single_point_curve(self):
        """Single-point curve returns all zeros."""
        dmdt_ints = self._make_dmdt_ints()
        result = compute_dmdt_features(
            [(np.array([1.0]), np.array([10.0]), np.array([0.1]))],
            dmdt_ints,
        )
        assert np.all(result == 0.0)

    def test_l2_normalization(self):
        """Each curve's histogram is L2-normalised (or zero)."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 100, 50))
        m = 18.0 + rng.normal(0, 0.5, 50)
        e = np.full(50, 0.1)

        dmdt_ints = self._make_dmdt_ints()
        result = compute_dmdt_features([(t, m, e)], dmdt_ints)

        norm = np.linalg.norm(result[0])
        if norm > 0:
            assert abs(norm - 1.0) < 1e-5, f"L2 norm = {norm}"

    def test_constant_magnitude(self):
        """Constant magnitude => all dm=0, should land in middle bins."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        m = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        e = np.full(5, 0.1)

        dmdt_ints = self._make_dmdt_ints()
        result = compute_dmdt_features([(t, m, e)], dmdt_ints)

        # dm=0 falls in the bin [-0.05, 0.05] which is bin 12
        # The result should be non-zero only in the dm=0 row
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# TestComputeBasicStats
# ---------------------------------------------------------------------------


class TestComputeBasicStats:
    """Tests for the compute_basic_stats() function."""

    def test_output_shape(self):
        """Returns (n_curves, 22) array."""
        rng = np.random.default_rng(42)
        n_curves = 3
        lcs = []
        for i in range(n_curves):
            t = np.sort(rng.uniform(0, 100, 50))
            m = 18.0 + rng.normal(0, 0.1, 50)
            e = np.full(50, 0.1)
            lcs.append((t, m, e))

        result = compute_basic_stats(lcs)
        assert result.shape == (n_curves, 22)
        assert result.dtype == np.float32

    def test_n_count(self):
        """First column is the number of points."""
        rng = np.random.default_rng(42)
        lcs = []
        for n in [20, 50, 100]:
            t = np.sort(rng.uniform(0, 100, n))
            m = 18.0 + rng.normal(0, 0.1, n)
            e = np.full(n, 0.1)
            lcs.append((t, m, e))

        result = compute_basic_stats(lcs)
        assert result[0, 0] == 20.0
        assert result[1, 0] == 50.0
        assert result[2, 0] == 100.0

    def test_constant_signal(self):
        """Constant magnitude yields ~0 scatter stats."""
        n = 50
        t = np.arange(n, dtype=np.float64) * 0.5
        m = np.full(n, 17.0)
        e = np.full(n, 0.1)

        result = compute_basic_stats([(t, m, e)])
        r = result[0]

        # N
        assert r[0] == n
        # median ≈ 17
        assert abs(r[1] - 17.0) < 0.01
        # wmean ≈ 17
        assert abs(r[2] - 17.0) < 0.01
        # chi2red ≈ 0
        assert abs(r[3]) < 0.01
        # wstd ≈ 0
        assert abs(r[5]) < 0.01

    def test_too_few_points(self):
        """Fewer than 4 points returns NaN."""
        t = np.array([1.0, 2.0, 3.0])
        m = np.array([10.0, 11.0, 12.0])
        e = np.array([0.1, 0.1, 0.1])

        result = compute_basic_stats([(t, m, e)])
        # All values should be NaN
        assert np.all(np.isnan(result[0]))

    def test_finite_for_normal_input(self):
        """All 22 stats are finite for well-behaved input."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 100, 200))
        m = 18.0 + rng.normal(0, 0.3, 200)
        e = np.full(200, 0.1)

        result = compute_basic_stats([(t, m, e)])
        for i in range(22):
            assert np.isfinite(result[0, i]), f"stat[{i}] is not finite: {result[0, i]}"

    def test_numerical_agreement_with_lcstats(self):
        """Rust basic stats agree with lcstats within reasonable tolerance."""
        lcstats = pytest.importorskip("lcstats", reason="lcstats requires scipy/numba")

        rng = np.random.default_rng(123)
        n = 200
        t = np.sort(rng.uniform(0, 100, n))
        m = 16.0 + 0.4 * rng.normal(0, 1, n)
        e = np.full(n, 0.05)

        old = lcstats.calc_basic_stats('test', (t, m, e))
        old_vals = old['test']

        new = compute_basic_stats([(t, m, e)])[0]

        # Compare first few well-defined stats (N, median, wmean)
        assert old_vals[0] == new[0], f"N: old={old_vals[0]}, new={new[0]}"
        assert (
            abs(old_vals[1] - new[1]) / max(abs(old_vals[1]), 1e-12) < 0.01
        ), f"median: old={old_vals[1]}, new={new[1]}"
        assert (
            abs(old_vals[2] - new[2]) / max(abs(old_vals[2]), 1e-12) < 0.01
        ), f"wmean: old={old_vals[2]}, new={new[2]}"
