# 2.2 Homogenization

Scripts, change logs, and test outputs for the quality control and homogenization of the station data (Section 2.2 of the paper).

Each station series was tested with the relative Standard Normal Homogeneity Test (SNHT) on the monthly station-minus-reference anomaly, using ERA5 at the nearest grid point (35.9°N, 14.5°E) as the reference. In rounds 3–10, MERRA-2 (36.0°N, 14.375°E) was added as an independent second reference, and the 5% critical values were replaced with those of Khaliq and Ouarda (2007); rounds 1–2 used a table close to the 1% level.

## How the scripts were run

- All scripts sit in `scripts/` and import paths, loaders, the SNHT, and the critical values from `snht_TS_ERA5.py`.
- Rounds 1–2: correction scripts never overwrite their input. Each writes `Daily_<station>_corrected.csv`, which then replaced `Daily_<station>.csv` before the next step, so every step reads the output of the previous one. After each step, `snht_TS_ERA5.py` was rerun to check the result.
- Rounds 3–10: `apply_residual_fixes.py` applies the fixes in place to the files in `Data`, starting from the v2 data (rounds 1–2 output).
- GitHub holds one copy of each script, set to the configuration of its last run. The table below gives the configuration of every run, so any step can be reproduced by setting those values. "Local copy" is the per-run copy kept during the work.
- Every adjustment is listed in `change_logs/Changes_summary.csv` (rounds 1–2), `change_logs/residual_fixes_summary.csv` (round 3), and `residual_fixes_summary_r4.csv` to `_r8.csv` (rounds 4–8).

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
| – | `era5_minmax_monthly.py` | same | ERA5 daily maximum and minimum 2 m temperature (CDS daily statistics); download stopped (TMX 2007–2024, TMN 2007 only) and never averaged to monthly, so not used | none |
| 17 | `apply_set_missing.py` | `apply_set_missing_temp4-luqa.py` | Luqa TAV 2015-10-01 to 2015-11-30, then 2015-09-25 to 2015-11-30 (cold drift before the December 2015 failure); two runs | `17_temp4-luqa.log` |
| 18 | `homogenize_segments.py` | `homogenize_segments-temp5-giordan.py` | Giordan TAV, additive; segments 2007-01 to 2008-12, 2011-01 to 2014-07, 2014-08 to 2022-12; reference: third | `18_temp5-giordan.log` |
| 19 | `homogenize.py` | `homogenize-ws1.py` | WS, additive, min_seg 18; Valletta anchor last, Msida anchor first | `19_ws1.log` |
| 20 | `homogenize_segments.py` | `homogenize_segments-ws2-xaghra.py`, `homogenize_segments-ws2-selmun.py` | WS, multiplicative; Xagħra segments 2008-01 to 2011-12, 2012-07 to 2015-12, 2016-01 to 2025-12; Selmun 2008-01 to 2011-12, 2012-06 to 2015-12, 2016-04 to 2022-06; reference: third; two runs | `20_ws2.log` |
| 21 | `flag_solar_outliers.py` | same | QUANTILE 0.99, MARGIN 1.10, SMOOTH 10; dry run, then APPLY True; 6 Luqa values, spring 2019 | `21_solar4-outliers.log` |
| 22 | `apply_corrections.py` | same | TMX and TMN inherit the TAV adjustments (steps 13–18); gust inherits WS (steps 19–20); RESET_FROM_RAW True | `22_max-min-gust.log` |
| 23 | `wind_direction_qc.py`, `flag_low_speed_directions.py`, `wind_dir_corruption_check.py`, `plot_windrose_period_panel.py`, `apply_corrections.py` | same | Wind direction (QC only, not homogenized). (a) Directions of 0° during calm (≤0.5 m/s) set missing. (b) 2013–2015 directions with wind speed ≤2.0 m/s set missing (listed in `wind_dir_flags.csv`); this reduced but did not remove the bias. (c) Per-station check of directions falling in sectors rare in the good years; (d) wind roses for 2007–2012, 2013–2015, and 2016–2025. (e) 2013-01-01 to 2015-12-31 set missing at XR, XW, SL, DN, VL, LQ, and BN (northward vane bias, also in strong wind); GD unchanged | none (rows in `Changes_summary.csv`) |
| 24 | `snht_TS_ERA5.py` | same | Final test of the homogenized data | `v2_homogenized/`, `Fig2_snht_era5_summary.png` |

A copy of `apply_step_fix.py` set to a Luqa pressure break at 2012-03 was prepared after step 3 but not used: step 4 found that break together with a second one, and `04_pressure-luqa2.log` starts from the T0 left by step 3 (27.5).

**Parameter choices in `homogenize.py` and `homogenize_steps.py`.** `mode`: additive for offsets of fixed size (pressure, temperature, RH clipped to 0–100%, and the Valletta and Msida wind-speed steps, where log-space estimates against ERA5's noisy wind were unstable), multiplicative for errors proportional to the value (solar, and the Xagħra and Selmun wind-speed depression in `homogenize_segments.py`). `min_seg_months`: the shortest segment the break search may create, which stops SNHT splitting off short segments at the record ends, where it is over-sensitive; 24 for pressure, 18 by default, and 12 where a genuine feature was shorter (a drift, or a tail of 12–24 months). Shorter anomalies at the record ends were set to missing instead. `anchor`: the segment all others are aligned to, normally the most recent (`last`, the current sensor); `first` was used where the latest segment was the faulty one (Msida WS, Dingli RH tail). Segments are aligned to each other, not to the reference, so each station keeps its own offset from ERA5.

### Rounds 3–10 (November 2026, revision)

| Step | Script | Configuration | Output |
|---|---|---|---|
| 25 | `download_merra2.sh`, `merge_merra2.py` | MERRA-2 M2SDNXSLV (daily), M2TMNXSLV and M2TMNXRAD (monthly), 2007–2025 | `Data/MERRA2/` |
| 26 | `snht_TS_ERA5.py` | v2 data re-tested with critical values from Khaliq and Ouarda (2007), 5%; shift per break; TMX and TMN tested against ERA5 mean 2 m temperature anomalies (the ERA5 daily extremes download was incomplete) | `v3_corrected-CV/` |
| 27 | `snht_TS_MERRA2.py` | Same test against MERRA-2, with TMX and TMN against MERRA-2 daily extremes | `v3_corrected-CV/` |
| 28 | `apply_residual_fixes.py` | `python apply_residual_fixes.py 3-10`: rounds 3 to 10 in order, from the v2 data (`FIXES[3]` to `FIXES[10]`; 24, 9, 6, 1, 2, 2, 7, and 1 fixes) | `23_residual_fixes.log` to `30_residual_fixes_r10.log`, `residual_fixes_summary*.csv` |
| 29 | `snht_TS_ERA5.py`, `snht_TS_MERRA2.py` | Final test | `v4_homogenized/` |

Each round's fixes come from re-testing the output of the previous round with both SNHT scripts; only the final test is kept. Each SNHT run reports only the largest break per series, so removing it can reveal the next.

Rule: every break significant against ERA5, the reference used throughout, is corrected, as is a break significant against MERRA-2 where ERA5 places its largest T(k) within two months (the same break). Break months and offsets come from ERA5; where both references are significant at different months, the ERA5 break is corrected. Other MERRA-2-only breaks are corrected with MERRA-2's month and offset unless the correction produces a significant break against ERA5, meaning the two references contradict each other. A trial correction showed this for Bengħajsa, Dingli, and Valletta RH (ERA5 T0 15.2, 18.1, and 26.2 afterwards), so these are reported, not corrected. Luqa pressure is treated the same way: MERRA-2 shows a +0.21 hPa step from 2021-06, the first of its reprocessed MERRA2_401 months (June to September 2021), while ERA5, which tracks Luqa pressure closely, shows none. Segments shorter than 24 months are set to missing: at the record ends, and between two breaks of opposite sign found in successive rounds. Rounds stop when no break meets the rule.

The script holds only the corrections that were kept, and all rounds were re-run in one pass from the v2 data. Corrections tried during the revision and withdrawn:
- MERRA-2-only breaks in Bengħajsa, Dingli, and Valletta RH, whose correction produced new significant breaks against ERA5 (see the rule).
- Step corrections for Giordan RH (2014–2015) and Xewkija TMX (2021). Later re-tests showed short anomalous segments instead (2014-02 to 2015-01, and 2021-02 to 2021-08), which are now set to missing.
- Giordan TAV and TMX adjusted for a break at 2021-11, which leaves only 13 months before the record ends (2022-12). These months are now set to missing.

## Folders

- `scripts/`: all scripts above.
- `change_logs/`: console output of each correction run, numbered in run order, plus the adjustment summaries.
- `v1_originals/`: SNHT results and figures for the original data.
- `v2_homogenized/`: SNHT results after rounds 1–2 (critical values close to the 1% level).
- `v3_corrected-CV/`: the v2 data re-tested at the corrected 5% critical values against ERA5 and MERRA-2; shows the residual breaks that rounds 3–10 remove.
- `v4_homogenized/`: SNHT results against ERA5 and MERRA-2 for the final data.

## References

Khaliq, M.N., and Ouarda, T.B.M.J. (2007). On the critical values of the standard normal homogeneity test (SNHT). *International Journal of Climatology*, 27, 681–687. doi:10.1002/joc.1438
