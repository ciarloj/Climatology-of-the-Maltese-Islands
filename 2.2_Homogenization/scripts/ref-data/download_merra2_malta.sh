mkdir -p merra2/slv
while read -r url || [ -n "$url" ]; do
  d=$(echo "$url" | grep -oE '\.[0-9]{8}\.nc4' | head -1 | tr -dc '0-9')
  f="merra2/slv/M2SDNXSLV_$d.nc4"
  [ -s "$f" ] && continue
  wget -q --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies \
       --keep-session-cookies --auth-no-challenge=on -O "$f" "$url" || rm -f "$f"
done < subset_links_895cf0b4-0943-4861-a280-9c8e5f6bcf38.txt
ls merra2/slv | wc -l   # should print 6940
