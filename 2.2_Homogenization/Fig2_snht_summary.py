"""
Fig2_snht_summary.py
--------------------
Fig. 2: SNHT summary of the final homogenized data, (a) against ERA5 and
(b) against MERRA-2. Cells show T0 (top) and the most likely break month
(bottom); a red outline marks T0 above the 5 % critical value. Both panels
share one colour scale.

USAGE (from 2.2_Homogenization/scripts)
    python Fig2_snht_summary.py [<era5_results.csv> <merra2_results.csv>]
Defaults: the results in ../v4_homogenized/figures_era5 and figures_merra2.

OUTPUT
    ../Fig2_snht_summary.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(HERE, "../../analysis/SNHT", "v13_residual-fixes_v10_homogenized")
ERA5_CSV = sys.argv[1] if len(sys.argv) > 2 else os.path.join(V4, "figures_era5", "snht_era5_results.csv")
MERRA_CSV = sys.argv[2] if len(sys.argv) > 2 else os.path.join(V4, "figures_merra2", "snht_merra2_results.csv")
OUT = os.path.join(HERE, "..", "Fig2_snht_summary.png")

VARIABLES = ["Temperature", "Max Temperature", "Min Temperature", "Relative Humidity",
             "Solar", "Pressure", "Wind Speed"]


def matrices(csv):
    df = pd.read_csv(csv)
    stations = sorted(df["Station"].unique())
    t0 = pd.DataFrame(np.nan, index=stations, columns=VARIABLES)
    yr = pd.DataFrame("", index=stations, columns=VARIABLES)
    sig = pd.DataFrame(False, index=stations, columns=VARIABLES)
    for _, r in df.iterrows():
        if r["Variable"] in VARIABLES:
            t0.loc[r["Station"], r["Variable"]] = r["T0"]
            yr.loc[r["Station"], r["Variable"]] = str(r["Break year"])
            sig.loc[r["Station"], r["Variable"]] = r["Significant"] == "YES"
    return t0, yr, sig


def panel(ax, t0, yr, sig, vmax, title):
    im = ax.imshow(t0.values.astype(float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    for i, s in enumerate(t0.index):
        for j, v in enumerate(t0.columns):
            val = t0.loc[s, v]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="grey")
                continue
            colour = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i - 0.13, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=colour)
            ax.text(j, i + 0.22, yr.loc[s, v], ha="center", va="center", fontsize=7.5, color=colour)
            if sig.loc[s, v]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="crimson", lw=2.5))
    ax.set_xticks(range(len(t0.columns)))
    ax.set_xticklabels(t0.columns, fontsize=10)
    ax.set_yticks(range(len(t0.index)))
    ax.set_yticklabels(t0.index, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    return im


def main():
    e = matrices(ERA5_CSV)
    m = matrices(MERRA_CSV)
    vmax = np.nanmax([np.nanmax(e[0].values), np.nanmax(m[0].values)])

    fig, axes = plt.subplots(2, 1, figsize=(15, 13))
    panel(axes[0], *e, vmax, "(a) ERA5")
    im = panel(axes[1], *m, vmax, "(b) MERRA-2")
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("T₀", fontsize=11)

    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
