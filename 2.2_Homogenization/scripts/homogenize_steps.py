"""
homogenize_steps.py
-------------------
Automated multi-breakpoint ADDITIVE homogenisation against ERA5.

Replaces the manual "detect -> fix -> re-run -> fix" loop. In one run it:
  1. detects ALL step breaks via binary segmentation (SNHT, each split must
     exceed the critical value and leave segments >= MIN_SEG months),
  2. adjusts every segment to a common anchor level,
  3. verifies no break above the critical value remains,
  4. applies the per-segment offsets to the RAW daily series and writes the
     corrected file, preserving corrections already made to OTHER columns.

For ADDITIVE steps only (pressure, temperature). Not for multiplicative
(solar) or set-to-missing cases.

Reads RAW data for the target variable, so re-running is idempotent (it never
double-applies). Place in the same folder as snht_TS_ERA5.py, then:
    python homogenize_steps.py
"""
import os
import pandas as pd
from snht_TS_ERA5 import (load_all_stations, load_era5_reference, snht,
                          critical_value, DATA_DIR)

# ----------------------------- CONFIG --------------------------------------
STATION  = "Luqa"
VARIABLE = "Pressure"
RAW_FILE = f"Daily_{STATION}.csv"
MIN_SEG  = 24          # minimum months per segment (guards against spurious end-breaks)
ANCHOR   = "last"      # segment that defines the kept level: "last", "first", or "longest"
MISSING  = "-"
# ---------------------------------------------------------------------------


def find_breaks(diff, min_seg):
    """Binary segmentation: sorted break Periods where SNHT T0 exceeds the CV."""
    breaks = []
    def rec(s):
        if len(s) < 2 * min_seg:
            return
        _, k0, T0 = snht(s.values)
        if T0 <= critical_value(len(s)):
            return
        bp = s.index[k0 - 1]
        left, right = s[s.index <= bp], s[s.index > bp]
        if len(left) < min_seg or len(right) < min_seg:
            return
        breaks.append(bp)
        rec(left)
        rec(right)
    rec(diff.dropna())
    return sorted(breaks)


def segments_from_breaks(breaks):
    """List of (start_excl, end_incl) Period bounds; None = open end."""
    segs, prev = [], None
    for b in breaks:
        segs.append((prev, b))
        prev = b
    segs.append((prev, None))
    return segs


def seg_mask(index_periods, a, b):
    mm = pd.Series(True, index=index_periods)
    if a is not None:
        mm &= index_periods > a
    if b is not None:
        mm &= index_periods <= b
    return mm


def main():
    stations = load_all_stations()
    era5 = load_era5_reference()
    cand = stations[STATION][VARIABLE].dropna()
    ref = era5[VARIABLE].dropna()
    common = cand.index.intersection(ref.index)
    diff = (cand.loc[common] - ref.loc[common]).dropna()

    print("=" * 64)
    print(f"MULTI-BREAK HOMOGENISATION  --  {STATION} / {VARIABLE}")
    print("=" * 64)

    breaks = find_breaks(diff, MIN_SEG)
    if not breaks:
        print("No breaks above the critical value. Nothing to do.")
        return
    print(f"Breaks detected ({len(breaks)}): " + ", ".join(str(b) for b in breaks))

    segs = segments_from_breaks(breaks)
    seg_means = [diff[seg_mask(diff.index, a, b)].mean() for a, b in segs]
    seg_lens = [int(seg_mask(diff.index, a, b).sum()) for a, b in segs]

    if ANCHOR == "first":
        target = seg_means[0]
    elif ANCHOR == "longest":
        target = seg_means[seg_lens.index(max(seg_lens))]
    else:  # "last"
        target = seg_means[-1]

    print(f"\nSegment levels (mean station-ERA5 diff) and offset applied "
          f"(anchor = {ANCHOR}):")
    offsets = []
    for (a, b), mu, n in zip(segs, seg_means, seg_lens):
        off = target - mu
        offsets.append((a, b, off))
        la = "start" if a is None else str(a + 1)
        lb = "end" if b is None else str(b)
        print(f"  {la:>9s} .. {lb:<9s}  n={n:3d}  mean={mu:+.2f}  offset={off:+.2f}")

    # ---- verify on the monthly difference ----
    dcorr = diff.copy()
    for a, b, off in offsets:
        dcorr[seg_mask(dcorr.index, a, b)] += off
    cv = critical_value(len(diff))
    resid = find_breaks(dcorr, MIN_SEG)
    print(f"\nSNHT T0  before={snht(diff.values)[2]:.1f}  "
          f"after={snht(dcorr.values)[2]:.1f}  CV={cv:.1f}")
    if resid:
        print(f"WARNING residual breaks remain: {resid}")
        print("        -> lower MIN_SEG, or inspect (offsetting/cancelling breaks).")
    else:
        print("VERIFICATION PASSED -- no residual break above the critical value.")

    # ---- apply offsets to RAW daily, preserve other columns' corrections ----
    raw_path = os.path.join(DATA_DIR, RAW_FILE)
    out_path = os.path.join(DATA_DIR, f"Daily_{STATION}_corrected.csv")
    base_path = out_path if os.path.exists(out_path) else raw_path

    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    base = pd.read_csv(base_path, dtype=str, keep_default_na=False)
    if not (raw["Date"] == base["Date"]).all():
        raise SystemExit("Date columns of raw and corrected files do not align.")
    dts_per = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.to_period("M")

    row_off = pd.Series(0.0, index=raw.index)
    for a, b, off in offsets:
        m = pd.Series(True, index=raw.index)
        if a is not None:
            m &= dts_per > a
        if b is not None:
            m &= dts_per <= b
        row_off[m] = off

    col = raw[VARIABLE]
    num = pd.to_numeric(col.where(~col.isin(["", MISSING])), errors="coerce")
    corr = num + row_off
    fmt = corr.map(lambda x: "" if pd.isna(x) else f"{x:.2f}".rstrip("0").rstrip("."))
    apply_rows = (row_off != 0) & num.notna()          # only reformat genuinely-changed cells
    out_col = col.copy()
    out_col[apply_rows] = fmt[apply_rows]
    base[VARIABLE] = out_col
    base.to_csv(out_path, index=False, na_rep=MISSING)
    print(f"\nApplied offsets to {int(apply_rows.sum())} daily '{VARIABLE}' rows "
          f"(anchor segment unchanged); wrote {out_path}")
    print("Original file left unchanged.")


if __name__ == "__main__":
    main()
