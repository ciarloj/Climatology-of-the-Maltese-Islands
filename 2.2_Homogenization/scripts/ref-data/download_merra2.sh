#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download MERRA-2 OPeNDAP subset files from a GES DISC links list (.txt).
#
# Usage:   bash download_merra2.sh <links.txt> <output_dir>
# Example: bash download_merra2.sh subset_links_XXXX.txt merra2/slv
#
# - Files are named <COLLECTION>_<date>.nc4, e.g. M2SDNXSLV_20070101.nc4 (daily)
#   or M2TMNXSLV_200701.nc4 (monthly), so the same script works for every list.
# - Re-run it to resume: files already downloaded are skipped.
# - Nothing is deleted. A failed download (empty file or an Earthdata login
#   page instead of data) is moved to <output_dir>/failed/ for inspection,
#   and is retried on the next run.
# - Needs ~/.netrc with your Earthdata login, and ~/.urs_cookies.
# ---------------------------------------------------------------------------

links="$1"
outdir="$2"
if [ -z "$links" ] || [ -z "$outdir" ] || [ ! -f "$links" ]; then
    echo "Usage: bash $0 <links.txt> <output_dir>"
    exit 1
fi
mkdir -p "$outdir"

total=$(grep -c '\.nc4' "$links")
count=0; new=0; had=0; failed=0
coll=""

while read -r url || [ -n "$url" ]; do
    [ -z "$url" ] && continue
    coll=$(echo "$url" | grep -oE 'granules/[A-Z0-9]+' | head -1 | cut -d/ -f2)
    d=$(echo "$url" | grep -oE '\.[0-9]{6,8}\.nc4' | head -1 | cut -d. -f2)
    f="$outdir/${coll}_${d}.nc4"

    if [ -s "$f" ]; then
        had=$((had + 1))
    else
        wget -q --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies \
             --keep-session-cookies --auth-no-challenge=on -O "$f" "$url"
        # A real file is netCDF-4 (HDF5): its first bytes contain "HDF"
        if [ -s "$f" ] && head -c 8 "$f" | grep -q "HDF"; then
            new=$((new + 1))
        else
            mkdir -p "$outdir/failed"
            mv -f "$f" "$outdir/failed/" 2>/dev/null
            failed=$((failed + 1))
        fi
    fi

    count=$((count + 1))
    printf "\r[%d/%d]  new: %d  |  already had: %d  |  failed: %d " \
           "$count" "$total" "$new" "$had" "$failed"
done < "$links"
echo

present=$(ls "$outdir"/"${coll}"_*.nc4 2>/dev/null | wc -l)
echo "Files present in $outdir: $present of $total expected"
echo

if [ "$present" -eq "$total" ]; then
cat <<'EOF'
       /\_/\
      ( ^.^ )   All files present. Purrfect.
       > ^ <
      /     \
     (  | |  )
      \_|_|_/~
EOF
else
cat <<EOF
       /\_/\\
      ( o.o )   $((total - present)) file(s) missing. Run me again to retry
       > ^ <    (failed downloads are in $outdir/failed/).
EOF
fi
