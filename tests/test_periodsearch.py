"""Tests for scope-ml period searching using the periodfind unified device API.

Uses synthetic lightcurve data — no GPU or Kowalski needed.

Run with:
    python -m pytest tests/test_periodsearch.py -v
"""

import numpy as np

import periodfind

import sys
import os

from periodsearch import (
    find_periods,
    _normalize_algorithm,
    _prepare_lightcurves,
    _build_pdots,
)

# Add the tools directory to the path so we can import periodsearch
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'featureGeneration')
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

    def test_gpu_algorithm_names(self):
        """ECE/EAOV/ELS names work with doCPU=True (maps to same underlying algo)."""
        lcs = [make_sinusoidal_lightcurve(period=3.0)]
        freqs = make_freq_grid(0.1, 2.0, 100)

        for algo_name in ["ECE", "EAOV", "ELS"]:
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
        )

        periodfind.set_device('cpu')
        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)
        aov = periodfind.AOV(n_phase=10)
        ls = periodfind.LombScargle()
        assert isinstance(ce, CpuCE)
        assert isinstance(aov, CpuAOV)
        assert isinstance(ls, CpuLS)

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

    def test_e_prefix(self):
        assert _normalize_algorithm("ECE") == ("CE", False)
        assert _normalize_algorithm("EAOV") == ("AOV", False)
        assert _normalize_algorithm("ELS") == ("LS", False)

    def test_periodogram_suffix(self):
        assert _normalize_algorithm("CE_periodogram") == ("CE", True)
        assert _normalize_algorithm("ECE_periodogram") == ("CE", True)
        assert _normalize_algorithm("EAOV_periodogram") == ("AOV", True)

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
