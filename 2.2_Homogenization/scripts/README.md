# 2.2 Homogenization

Scripts, change logs, and test outputs for the quality control and homogenization of the station data (Section 2.2 of the paper).

Each station series was tested with the relative Standard Normal Homogeneity Test (SNHT) on the monthly station-minus-reference anomaly, using ERA5 at the nearest grid point (35.9°N, 14.5°E) as the reference. In round 3, MERRA-2 (36.0°N, 14.375°E) was added as an independent second reference, and the 5% critical values were replaced with those of Khaliq and Ouarda (2007); rounds 1–2 used a table close to the 1% level.

## How the scripts were run

- All scripts sit in `scripts/` and import paths, loaders, the SNHT, and the critical values from `snht_TS_ERA5.py`.
- Rounds 1–2: correction scripts never overwrite their input. Each writes `Daily_<station>_corrected.csv`, which then replaced `Daily_<station>.csv` before the next step, so every step reads the output of the previous one. After each step, `snht_TS_ERA5.py` was rerun to check the result.
- Round 3: `apply_residual_fixes.py` writes a complete copy of the dataset (`Data_v3`) and leaves `Data` untouched.
- GitHub holds one copy of each script, set to the configuration of its last run. The table below gives the configuration of every run, so any step can be reproduced by setting those values. "Local copy" is the per-run copy kept during the work.
- Every adjustment is listed in `change_logs/Changes_summary.csv` (rounds 1–2) and `change_logs/residual_fixes_summary.csv` (round 3).

## Run order

### Detection

| Step | Script | Configuration | Output |
|---|---|---|---|
| D1 | `download_era5.py` | ERA5 monthly means, 2007–2025 | `Data/ERA5/era5_monthly_YYYY.nc` |
| D2 | `snht_TS_ERA5.py` | All stations and variables against ERA5 | `v1_originals/` |

An earlier version, `snht_TS.py`, tested each station against the other stations. It was set aside because several breaks are synchronous across the network (notably at the 2015/16 boundary) and would contaminate a station-based reference.

### Rounds 1–2 (June 2026)

| Step | Script | Local copy | Configuration | Log |
|---|---|---|---|---|
| 1 | `apply_step_fix.py` | `apply_step_fix_pressure-xewkija.py` | Xewkija Pressure, BREAK 2015-11; −11.87 hPa from 2015-12 | `01_pressure-xewkija.log` |
| 2 | `apply_set_missing.py` | `apply_set_missing_solar1-msida-luqa.py` | Solar set missing: Msida 2012-12-21 to 2013-12-30; Luqa 2014-12-22 to 2015-12-31 (dead pyranometer) | `02_solar1.log` |
| 3 | `apply_step_fix.py` | `apply_step_fix_pressure-luqa.py` | Luqa Pressure, BREAK 2015-11; −1.14 hPa from 2015-12 (residual break remained) | `03_pressure-luqa1.log` |
| – | `apply_step_fix.py` | `apply_step_fix_pressure-luqa2.py` | TBC | none |
| 4 | `homogenize_steps.py` | `homogenize_steps-pressure.py` | Luqa Pressure, MIN_SEG 24, ANCHOR last; breaks 2012-03 and 2019-01 | `04_pressure-luqa2.log` |
| 5 | `homogenize.py` | `homogenize-solar.py` | Solar, multiplicative, anchor last; Luqa (min_seg 18), Xagħra (12), Msida (18) | `05_solar2.log` |
| 6 | `apply_set_missing.py` | `apply_set_missing_solar3-msida.py` | Msida Solar 2007-01-01 to 2007-08-31 (record-start spike, too short to scale) | `06_solar3.log` |
| 7 | `rh_saturation_scan.py` | same | THRESH 99.5%, FLAG_FRAC 0.15; report only | `07_rh0-sat-scan.log` |
| 8 | `apply_set_missing.py` | `apply_set_missing_rh1.py` | RH saturated spans: BN 2019-03 to 2021-11; SL 2018-01 to 2022-12; VL 2018-04 to 2018-06; XR 2017-08 to 2020-12; XW 2019-03 to 2022-09 | `08_rh1.log` |
| 9 | `homogenize.py` | `homogenize-rh2.py` | RH at all stations, additive, min_seg 18, anchor last, clip (0, 100) | `09_rh2.log` |
| 10 | `apply_set_missing.py` | `apply_set_missing_rh3-selmun.py` | Selmun RH 2017-07-01 to 2025-12-31 (jump at record end) | `10_rh3-selmun.log` |
| 11 | `homogenize.py` | `homogenize-rh3-dingli.py` | Dingli RH, additive, min_seg 12, anchor first, clip (0, 100) | `11_rh3-dingli.log` |
| 12 | `apply_set_missing.py` | `apply_set_missing_rh4-selmun.py` | Selmun RH 2014-07-01 to 2025-12-31 | `12_rh4-selmun.log` |
| 13 | `homogenize.py` | `homogenize-temp1.py` | TAV at all stations, additive, min_seg 18, anchor last | `13_temp1.log` |
| 14 | `homogenize.py` | `homogenize-temp2.py` | TAV at Luqa, Dingli, and Selmun, additive, min_seg 12, anchor last | `14_temp2a-homogenize.log` |
| 15 | `apply_set_missing.py` | `apply_set_missing_temp3-luqa.py` | Luqa TAV 2015-12-01 to 2016-01-31 (1–2 month cold transient) | `15_temp3a-luqa.log` |
| 16 | `fix_giordan_stub.py` | same | STUB 2007-01-01 to 2008-11-30, REF 2012-01-01 to 2014-12-31; +2.309 °C | `16_temp3b-giordan.log` |
| – | `era5_minmax_monthly.py` | same | ERA5 daily maximum and minimum 2 m temperature (CDS daily statistics), averaged to monthly | `era5_monthly_tmax.nc`, `era5_monthly_tmin.nc` |
| 17 | `apply_set_missing.py` | `apply_set_missing_temp4-luqa.py` | Luqa TAV 2015-10-01 to 2015-11-30, then 2015-09-25 to 2015-11-30 (cold drift before the December 2015 failure); two runs | `17_temp4-luqa.log` |
| 18 | `homogenize_segments.py` | `homogenize_segments-temp5-giordan.py` | Giordan TAV, additive; segments 2007-01 to 2008-12, 2011-01 to 2014-07, 2014-08 to 2022-12; reference: third | `18_temp5-giordan.log` |
| 19 | `homogenize.py` | `homogenize-ws1.py` | WS, additive, min_seg 18; Valletta anchor last, Msida anchor first | `19_ws1.log` |
| 20 | `homogenize_segments.py` | `homogenize_segments-ws2-xaghra.py`, `homogenize_segments-ws2-selmun.py` | WS, multiplicative; Xagħra segments 2008-01 to 2011-12, 2012-07 to 2015-12, 2016-01 to 2025-12; Selmun 2008-01 to 2011-12, 2012-06 to 2015-12, 2016-04 to 2022-06; reference: third; two runs | `20_ws2.log` |
| 21 | `flag_solar_outliers.py` | same | QUANTILE 0.99, MARGIN 1.10, SMOOTH 10; dry run, then APPLY True; 6 Luqa values, spring 2019 | `21_solar4-outliers.log` |
| 22 | `apply_corrections.py` | same | TMX and TMN inherit the TAV adjustments (steps 13–18); gust inherits WS (steps 19–20); RESET_FROM_RAW True | `22_max-min-gust.log` |
| 23 | TBC | TBC | Wind direction set missing 2013-01-01 to 2015-12-31 at XR, XW, SL, DN, VL, LQ, and BN (northward vane bias); GD unchanged | none (rows in `Changes_summary.csv`) |
| 24 | `snht_TS_ERA5.py` | same | Final test of the homogenized data | `v2_homogenized/`, `Fig2_snht_era5_summary.png` |

`README_homogenize_params.md` explains how `mode`, `min_seg_months`, and `anchor` were chosen for `homogenize.py`.

### Round 3 (November 2026, revision)

| Step | Script | Configuration | Output |
|---|---|---|---|
| 25 | `download_merra2.sh`, `merge_merra2.py` | MERRA-2 M2SDNXSLV (daily), M2TMNXSLV and M2TMNXRAD (monthly), 2007–2025 | `Data/MERRA2/` |
| 26 | `snht_TS_ERA5.py` | Critical values from Khaliq and Ouarda (2007), 5%; shift per break; TMX and TMN tested | `snht_era5_results.csv` |
| 27 | `snht_TS_MERRA2.py` | Same test against MERRA-2 | `snht_merra2_results.csv`, `snht_reference_comparison.csv` |
| 28 | `apply_residual_fixes.py` | 24 fixes listed in the script (FIXES); writes `Data_v3` | `23_residual_fixes.log`, `residual_fixes_summary.csv` |
| 29 | `snht_TS_ERA5.py`, `snht_TS_MERRA2.py` | Final test of the v3 data | `v3_homogenized/` |

Round 3 rule: a residual break is corrected when it is significant against at least one reference and the other reference places it within two months; segments at the record ends shorter than 24 months are set to missing.

## Folders

- `scripts/`: all scripts above.
- `change_logs/`: console output of each correction run, numbered in run order, plus the adjustment summaries.
- `v1_originals/`: SNHT results and figures for the original data.
- `v2_homogenized/`: SNHT results after rounds 1–2.
- `v3_homogenized/`: SNHT results against ERA5 and MERRA-2 after round 3.

## References

Khaliq, M.N., and Ouarda, T.B.M.J. (2007). On the critical values of the standard normal homogeneity test (SNHT). *International Journal of Climatology*, 27, 681–687. doi:10.1002/joc.1438
