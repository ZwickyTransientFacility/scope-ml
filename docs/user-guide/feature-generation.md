# Feature Generation

## Download IDs for ZTF Fields/CCDs/Quadrants

Create HDF5 file for single CCD/quad pair in a field:

```sh
get-quad-ids --catalog ZTF_source_features_DR16 --field 301 --ccd 2 --quad 3 --minobs 20 --skip 0 --limit 10000
```

Create multiple HDF5 files for some CCD/quad pairs in a field:

```sh
get-quad-ids --catalog ZTF_source_features_DR16 --field 301 --multi-quads --ccd-range 1 8 --quad-range 2 4 --minobs 20 --limit 10000
```

Create multiple HDF5 files for all CCD/quad pairs in a field:

```sh
get-quad-ids --catalog ZTF_source_features_DR16 --field 301 --multi-quads --minobs 20 --limit 10000
```

Create single HDF5 file for all sources in a field:

```sh
get-quad-ids --catalog ZTF_source_features_DR16 --field 301 --whole-field
```

## Download SCoPe Features for ZTF Fields/CCDs/Quadrants

First, run `get-quad-ids` for desired fields/CCDs/quads.

Download features for all sources in a field:

```sh
get-features --field 301 --whole-field
```

Download features for all sources in a field, imputing missing features using the strategies in `config.yaml`:

```sh
get-features --field 301 --whole-field --impute-missing-features
```

Download features for a range of CCD/quads individually:

```sh
get-features --field 301 --ccd-range 1 2 --quad-range 3 4
```

Download features for a single pair of CCD/quad:

```sh
get-features --field 301 --ccd-range 1 --quad-range 2
```

## Generating Features

Code has been adapted from [ztfperiodic](https://github.com/mcoughlin/ztfperiodic) and other sources to calculate basic and Fourier stats for light curves along with other features. This allows new features to be generated with SCoPe, both locally and using GPU cluster resources. The feature generation script is run using the `generate-features` command.

Currently, the basic stats are calculated via `tools/featureGeneration/lcstats.py`, and a host of period-finding algorithms are available in `tools/featureGeneration/periodsearch.py`. Among the CPU-based period-finding algorithms, there is not yet support for `AOV_cython`. For the `AOV` algorithm to work, run `source build.sh` in the `tools/featureGeneration/pyaov/` directory, then copy the newly created `.so` file (`aov.cpython-310-darwin.so` or similar) to `lib/python3.10/site-packages/` or equivalent within your environment. The GPU-based algorithms require CUDA support (so Mac GPUs are not supported).

### Inputs

| # | Argument | Description |
|---|----------|-------------|
| 1 | `--source-catalog`* | Name of Kowalski catalog containing ZTF sources (str) |
| 2 | `--alerts-catalog`* | Name of Kowalski catalog containing ZTF alerts (str) |
| 3 | `--gaia-catalog`* | Name of Kowalski catalog containing Gaia data (str) |
| 4 | `--bright-star-query-radius-arcsec` | Maximum angular distance from ZTF sources to query nearby bright stars in Gaia (float) |
| 5 | `--xmatch-radius-arcsec` | Maximum angular distance from ZTF sources to match external catalog sources (float) |
| 6 | `--limit` | Maximum number of sources to process in batch queries / statistics calculations (int) |
| 7 | `--period-algorithms`* | Dictionary containing names of period algorithms to run. Normally specified in config; if specified here, should be a list |
| 8 | `--period-batch-size` | Maximum number of sources to simultaneously perform period finding (int) |
| 9 | `--doCPU` | Flag to run config-specified CPU period algorithms (bool) |
| 10 | `--doGPU` | Flag to run config-specified GPU period algorithms (bool) |
| 11 | `--samples-per-peak` | Number of samples per periodogram peak (int) |
| 12 | `--doScaleMinPeriod` | For period finding, scale min period based on min-cadence-minutes (bool). Otherwise, set `--max-freq` to desired value |
| 13 | `--doRemoveTerrestrial` | Remove terrestrial frequencies from period-finding analysis (bool) |
| 14 | `--Ncore` | Number of CPU cores to parallelize queries (int) |
| 15 | `--field` | ZTF field to run (int) |
| 16 | `--ccd` | ZTF CCD to run (int) |
| 17 | `--quad` | ZTF quadrant to run (int) |
| 18 | `--min-n-lc-points` | Minimum number of points required to generate features for a light curve (int) |
| 19 | `--min-cadence-minutes` | Minimum cadence between light curve points. Higher-cadence data are dropped except for the first point in the sequence (float) |
| 20 | `--dirname` | Name of generated feature directory (str) |
| 21 | `--filename` | Prefix of each feature filename (str) |
| 22 | `--doCesium` | Flag to compute config-specified cesium features in addition to default list (bool) |
| 23 | `--doNotSave` | Flag to avoid saving generated features (bool) |
| 24 | `--stop-early` | Flag to stop feature generation before entire quadrant is run. Pair with `--limit` to run small-scale tests (bool) |
| 25 | `--doQuadrantFile` | Flag to use a generated file containing [jobID, field, ccd, quad] columns instead of specifying `--field`, `--ccd` and `--quad` (bool) |
| 26 | `--quadrant-file` | Name of quadrant file in the generated_features/slurm directory or equivalent (str) |
| 27 | `--quadrant-index` | Number of job in quadrant file to run (int) |
| 28 | `--doSpecificIDs` | Flag to perform feature generation for ztf_id column in config-specified file (bool) |
| 29 | `--skipCloseSources` | Flag to skip removal of sources too close to bright stars via Gaia (bool) |
| 30 | `--top-n-periods` | Number of (E)LS, (E)CE periods to pass to (E)AOV if using (E)LS_(E)CE_(E)AOV algorithm (int) |
| 31 | `--max-freq` | Maximum frequency [1/days] to use for period finding (float). Overridden by `--doScaleMinPeriod` |
| 32 | `--fg-dataset`* | Path to parquet, HDF5 or CSV file containing specific sources for feature generation (str) |
| 33 | `--max-timestamp-hjd`* | Maximum timestamp of queried light curves, HJD (float) |

**Output:** `feature_df` -- dataframe containing generated features.

\* Specified in `config.yaml`.

### Example Usage

The following is an example of running the feature generation script locally:

```sh
generate-features --field 301 --ccd 2 --quad 4 \
  --source-catalog ZTF_sources_20230109 \
  --alerts-catalog ZTF_alerts \
  --gaia-catalog Gaia_EDR3 \
  --bright-star-query-radius-arcsec 300.0 \
  --xmatch-radius-arcsec 2.0 \
  --query-size-limit 10000 \
  --period-batch-size 1000 \
  --samples-per-peak 10 \
  --Ncore 4 \
  --min-n-lc-points 50 \
  --min-cadence-minutes 30.0 \
  --dirname generated_features \
  --filename gen_features \
  --doCPU --doRemoveTerrestrial --doCesium
```

Setting `--doCPU` will run the config-specified CPU period algorithms on each source. Setting `--doGPU` instead will do likewise with the specified GPU algorithms. If neither of these keywords is set, the code will assign a value of `1.0` to each period and compute Fourier statistics using that number.

Below is an example using a job/quadrant file (containing [job id, field, ccd, quad] columns) instead of specifying field/ccd/quad directly:

```sh
generate-features \
  --source-catalog ZTF_sources_20230109 \
  --alerts-catalog ZTF_alerts \
  --gaia-catalog Gaia_EDR3 \
  --bright-star-query-radius-arcsec 300.0 \
  --xmatch-radius-arcsec 2.0 \
  --query-size-limit 10000 \
  --period-batch-size 1000 \
  --samples-per-peak 10 \
  --Ncore 20 \
  --min-n-lc-points 50 \
  --min-cadence-minutes 30.0 \
  --dirname generated_features_DR15 \
  --filename gen_features \
  --doGPU --doRemoveTerrestrial --doCesium \
  --doQuadrantFile --quadrant-file slurm.dat --quadrant-index 5738
```

### SLURM Scripts

For large-scale feature generation, `generate-features` is intended to be run on a high-performance computing cluster. Often these clusters require jobs to be submitted using a utility like SLURM (Simple Linux Utility for Resource Management) to generate scripts. These scripts contain information about the type, amount and duration of computing resources to allocate to the user.

SCoPe's `generate-features-slurm` code creates two SLURM scripts: (1) runs a single instance of `generate-features`, and (2) runs `generate-features-job-submission` which submits multiple jobs in parallel, periodically checking to see if additional jobs can be started.

`generate-features-slurm` can receive all of the arguments used by `generate-features`. These arguments are passed to the instances of feature generation begun by running SLURM script (1). There are also additional arguments specific to cluster resource management:

| # | Argument | Description |
|---|----------|-------------|
| 1 | `--job-name` | Name of submitted jobs (str) |
| 2 | `--cluster-name` | Name of HPC cluster (str) |
| 3 | `--partition-type` | Cluster partition to use (str) |
| 4 | `--nodes` | Number of nodes to request (int) |
| 5 | `--gpus` | Number of GPUs to request (int) |
| 6 | `--memory-GB` | Amount of memory to request in GB (int) |
| 7 | `--submit-memory-GB` | Memory allocation to request for job submission (int) |
| 8 | `--time` | Amount of time before instance times out (str) |
| 9 | `--mail-user` | User's email address for job updates (str) |
| 10 | `--account-name` | Name of account having HPC allocation (str) |
| 11 | `--python-env-name` | Name of Python environment to activate before running `generate_features.py` (str) |
| 12 | `--generateQuadrantFile` | Flag to map fields/CCDs/quads containing sources to job numbers, save file (bool) |
| 13 | `--field-list` | Space-separated list of fields for which to generate quadrant file. If None, all populated fields included (int) |
| 14 | `--max-instances` | Maximum number of HPC instances to run in parallel (int) |
| 15 | `--wait-time-minutes` | Amount of time to wait between status checks in minutes (float) |
| 16 | `--doSubmitLoop` | Flag to run loop initiating instances until out of jobs (bool) |
| 17 | `--runParallel` | Flag to run jobs in parallel using SLURM (recommended). Otherwise, run in series on a single instance (bool) |
| 18 | `--user` | If using SLURM, your username. This will be used to periodically run `squeue` and list your running jobs (str) |
| 19 | `--submit-interval-minutes` | Time to wait between job submissions, minutes (float) |

## Feature Definitions

### Selected Phenomenological Feature Definitions

| Name | Definition |
|------|------------|
| `ad` | Anderson-Darling statistic |
| `chi2red` | Reduced chi^2 after mean subtraction |
| `f1_BIC` | Bayesian information criterion of best-fitting series (Fourier analysis) |
| `f1_a` | a coefficient of best-fitting series (Fourier analysis) |
| `f1_amp` | Amplitude of best-fitting series (Fourier analysis) |
| `f1_b` | b coefficient of best-fitting series (Fourier analysis) |
| `f1_phi0` | Zero-phase of best-fitting series (Fourier analysis) |
| `f1_power` | Normalized chi^2 of best-fitting series (Fourier analysis) |
| `f1_relamp1` | Relative amplitude, first harmonic (Fourier analysis) |
| `f1_relamp2` | Relative amplitude, second harmonic (Fourier analysis) |
| `f1_relamp3` | Relative amplitude, third harmonic (Fourier analysis) |
| `f1_relamp4` | Relative amplitude, fourth harmonic (Fourier analysis) |
| `f1_relphi1` | Relative phase, first harmonic (Fourier analysis) |
| `f1_relphi2` | Relative phase, second harmonic (Fourier analysis) |
| `f1_relphi3` | Relative phase, third harmonic (Fourier analysis) |
| `f1_relphi4` | Relative phase, fourth harmonic (Fourier analysis) |
| `i60r` | Mag ratio between 20th, 80th percentiles |
| `i70r` | Mag ratio between 15th, 85th percentiles |
| `i80r` | Mag ratio between 10th, 90th percentiles |
| `i90r` | Mag ratio between 5th, 95th percentiles |
| `inv_vonneumannratio` | Inverse of Von Neumann ratio |
| `iqr` | Mag ratio between 25th, 75th percentiles |
| `median` | Median magnitude |
| `median_abs_dev` | Median absolute deviation of magnitudes |
| `norm_excess_var` | Normalized excess variance |
| `norm_peak_to_peak_amp` | Normalized peak-to-peak amplitude |
| `roms` | Root of mean magnitudes squared |
| `skew` | Skew of magnitudes |
| `smallkurt` | Kurtosis of magnitudes |
| `stetson_j` | Stetson J coefficient |
| `stetson_k` | Stetson K coefficient |
| `sw` | Shapiro-Wilk statistic |
| `welch_i` | Welch I statistic |
| `wmean` | Weighted mean of magnitudes |
| `wstd` | Weighted standard deviation of magnitudes |
| `dmdt` | Magnitude-time histograms (26x26) |

### Selected Ontological Feature Definitions

| Name | Definition |
|------|------------|
| `mean_ztf_alert_braai` | Mean significance of ZTF alerts for this source |
| `n_ztf_alerts` | Number of ZTF alerts for this source |
| `period` | Period determined by subscripted algorithms (e.g. ELS_ECE_EAOV) |
| `significance` | Significance of period |
| `AllWISE_w1mpro` | AllWISE W1 mag |
| `AllWISE_w1sigmpro` | AllWISE W1 mag error |
| `AllWISE_w2mpro` | AllWISE W2 mag |
| `AllWISE_w2sigmpro` | AllWISE W2 mag error |
| `AllWISE_w3mpro` | AllWISE W3 mag |
| `AllWISE_w4mpro` | AllWISE W4 mag |
| `Gaia_EDR3__parallax` | Gaia parallax |
| `Gaia_EDR3__parallax_error` | Gaia parallax error |
| `Gaia_EDR3__phot_bp_mean_mag` | Gaia BP mag |
| `Gaia_EDR3__phot_bp_rp_excess_factor` | Gaia BP-RP excess factor |
| `Gaia_EDR3__phot_g_mean_mag` | Gaia G mag |
| `Gaia_EDR3__phot_rp_mean_mag` | Gaia RP mag |
| `PS1_DR1__gMeanPSFMag` | PS1 g mag |
| `PS1_DR1__gMeanPSFMagErr` | PS1 g mag error |
| `PS1_DR1__rMeanPSFMag` | PS1 r mag |
| `PS1_DR1__rMeanPSFMagErr` | PS1 r mag error |
| `PS1_DR1__iMeanPSFMag` | PS1 i mag |
| `PS1_DR1__iMeanPSFMagErr` | PS1 i mag error |
| `PS1_DR1__zMeanPSFMag` | PS1 z mag |
| `PS1_DR1__zMeanPSFMagErr` | PS1 z mag error |
| `PS1_DR1__yMeanPSFMag` | PS1 y mag |
| `PS1_DR1__yMeanPSFMagErr` | PS1 y mag error |
