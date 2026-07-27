"""
homogenize.py
-------------
Unified ERA5-referenced homogenisation for ALL variables, in one engine.

Two modes, selected per job:
  * "additive"        (pressure, temperature): operate on   station - ERA5
  * "multiplicative"  (solar, ...):            operate on   log(station) - log(ERA5)
                                               so a proportional error becomes additive.

In BOTH modes the same pipeline runs: deseasonalise the difference, find ALL
breaks by binary segmentation (SNHT), adjust every segment onto one anchor,
verify nothing significant remains, apply to the daily series, and write the
corrected file (other columns preserved). Steps, boxes (e.g. Luqa solar) and
drifts (e.g. Xaghra solar, approximated as a staircase) are all handled by the
segmentation -- no per-case special-casing.

PIPELINE / IDEMPOTENCY
  Run order is:  apply_set_missing.py  (mask dead sensors)  ->  this script.
  Each job reads the station's *_corrected.csv if it exists (so set-missing and
  other variables' corrections are inherited), detects on that, and writes back.
  Because the correction removes the breaks, re-running a finished job finds
  nothing to do -- so it is idempotent in practice. Missing cells ("-") are
  never touched.

REQUIRES  load_era5_reference(raw=True)  -- the one-line change noted in chat.

Edit JOBS, then:  python homogenize.py
"""
import os
import numpy as np
import pandas as pd
from snht_TS_ERA5 import (load_era5_reference, snht, critical_value,
                          DATA_DIR, PERIOD_START, PERIOD_END)

MISSING = "-"

# ----------------------------- JOBS ----------------------------------------
# (station_file, variable, mode, min_seg_months, anchor[, clip])
#   anchor: "last" (current sensor, default), "first", or "longest"
#   clip:   optional (lo, hi) bounds for the corrected value -- use for RH = (0, 100)
# Run apply_set_missing.py FIRST (saturated/dead RH sectors), then these.
JOBS = [
    ("Daily_Valletta.csv", "Wind Speed", "additive", 18, "last"),
    ("Daily_Birkirkara.csv", "Wind Speed", "additive", 18, "first"),
]
# ---------------------------------------------------------------------------


def station_monthly(path, variable):
    """Raw monthly mean of one variable from a daily CSV (>=20 obs/month)."""
    df = pd.read_csv(path, na_values=["-", "N/A", "n/a", "NA", ""])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M")
    g = df.groupby("Month")[variable]
    m, c = g.mean(), g.count()
    m[c < 20] = np.nan
    return m.loc[PERIOD_START:PERIOD_END]


def deseasonalise(s):
    cal = pd.Series(s.index.month, index=s.index)
    clim = s.groupby(s.index.month).mean()
    return s - cal.map(clim)


def find_breaks(diff, min_seg):
    breaks = []
    def rec(s):
        s = s.dropna()
        if len(s) < 2 * min_seg:
            return
        _, k0, T0 = snht(s.values)
        if T0 <= critical_value(len(s)):
            return
        bp = s.index[k0 - 1]
        l, r = s[s.index <= bp], s[s.index > bp]
        if len(l) < min_seg or len(r) < min_seg:
            return
        breaks.append(bp)
        rec(l); rec(r)
    rec(diff.dropna())
    return sorted(breaks)


def segments(breaks):
    segs, prev = [], None
    for b in breaks:
        segs.append((prev, b)); prev = b
    segs.append((prev, None))
    return segs


def smask(idx, a, b):
    m = pd.Series(True, index=idx)
    if a is not None:
        m &= idx > a
    if b is not None:
        m &= idx <= b
    return m


def homogenize_one(station_file, variable, mode, min_seg, anchor, clip=None, era5=None):
    base = os.path.basename(station_file)
    out_name = base[:-4] + "_corrected.csv"
    out_path = os.path.join(DATA_DIR, out_name)
    src_path = out_path if os.path.exists(out_path) else os.path.join(DATA_DIR, base)

    print("=" * 64)
    print(f"{base}  /  {variable}   [{mode}]   (input: {os.path.basename(src_path)})")
    print("=" * 64)

    st = station_monthly(src_path, variable)
    e5 = era5[variable]
    common = st.dropna().index.intersection(e5.dropna().index)
    st, e5 = st.loc[common], e5.loc[common]

    if mode == "multiplicative":
        ok = (st > 0) & (e5 > 0)
        tdiff = np.log(st[ok]) - np.log(e5[ok])
    else:
        tdiff = st - e5
    diff = deseasonalise(tdiff).dropna()

    breaks = find_breaks(diff, min_seg)
    if not breaks:
        print("No breaks above the critical value. Nothing to do.")
        return
    print(f"Breaks detected ({len(breaks)}): " + ", ".join(str(b) for b in breaks))

    segs = segments(breaks)
    means = [diff[smask(diff.index, a, b)].mean() for a, b in segs]
    lens = [int(smask(diff.index, a, b).sum()) for a, b in segs]
    if anchor == "first":
        target = means[0]
    elif anchor == "longest":
        target = means[lens.index(max(lens))]
    else:
        target = means[-1]

    print(f"Segment adjustments (anchor={anchor}):")
    adj = []
    for (a, b), mu, n in zip(segs, means, lens):
        off = target - mu                       # in transform space
        adj.append((a, b, off))
        la = "start" if a is None else str(a + 1)
        lb = "end" if b is None else str(b)
        shown = f"x{np.exp(off):.3f}" if mode == "multiplicative" else f"{off:+.2f}"
        print(f"  {la:>9s} .. {lb:<9s}  n={n:3d}  level={mu:+.3f}  adjust={shown}")

    # verify on the monthly difference
    dc = diff.copy()
    for a, b, off in adj:
        dc[smask(dc.index, a, b)] += off
    cv = critical_value(len(diff))
    resid = find_breaks(dc, min_seg)
    msg = "VERIFICATION PASSED" if not resid else f"WARNING residual {resid} (lower min_seg / review)"
    print(f"SNHT T0  before={snht(diff.values)[2]:.1f}  after={snht(dc.values)[2]:.1f}  CV={cv:.1f}  -> {msg}")

    # ---- apply to the daily series, preserve everything else ----
    bdf = pd.read_csv(src_path, dtype=str, keep_default_na=False)
    per = pd.to_datetime(bdf["Date"], dayfirst=True, errors="coerce").dt.to_period("M")
    off_row = pd.Series(0.0, index=bdf.index)
    for a, b, off in adj:
        m = pd.Series(True, index=bdf.index)
        if a is not None:
            m &= per > a
        if b is not None:
            m &= per <= b
        off_row[m] = off

    col = bdf[variable]
    num = pd.to_numeric(col.where(~col.isin(["", MISSING])), errors="coerce")
    new = num * np.exp(off_row) if mode == "multiplicative" else num + off_row
    if clip is not None:                                   # bounded variables (e.g. RH -> [0,100])
        new = new.clip(clip[0], clip[1])
    fmt = new.map(lambda x: "" if pd.isna(x) else f"{x:.2f}".rstrip("0").rstrip("."))
    changed = (off_row != 0) & num.notna()
    out = col.copy()
    out[changed] = fmt[changed]
    bdf[variable] = out
    bdf.to_csv(out_path, index=False, na_rep=MISSING)
    print(f"Adjusted {int(changed.sum())} daily '{variable}' rows (missing untouched) -> {out_name}")


def main():
    era5 = load_era5_reference(raw=True)          # loaded once, reused for all jobs
    for job in JOBS:
        homogenize_one(*job, era5=era5)


if __name__ == "__main__":
    main()
