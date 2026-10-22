#!/usr/bin/env bash
# Download LungHist700 (Diosdado et al., Sci. Data 2024, CC BY 4.0) from figshare and lay it out as
# the project expects:
#   dataset/lunghist700_original/<aca_bd|...|nor>/*.jpg  + data.csv (image -> patient table)
#   dataset/ADC/*.jpg  (aca_bd + aca_md + aca_pd)      dataset/SCC/*.jpg  (scc_bd + scc_md + scc_pd)
# Idempotent: does nothing if the layout is already complete.
# Usage: bash scripts/fetch_lunghist700.sh
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${LUNGHIST700_URL:-https://ndownloader.figshare.com/files/45206104}"   # doi:10.6084/m9.figshare.25459174
ORIG=dataset/lunghist700_original

count() { find "$1" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' '; }
if [ -f "$ORIG/data.csv" ] && [ "$(count dataset/ADC)" = 280 ] && [ "$(count dataset/SCC)" = 260 ]; then
  echo "LungHist700 already in place (280 ADC, 260 SCC, data.csv)."; exit 0
fi

if ! command -v bsdtar >/dev/null; then
  echo "bsdtar (libarchive) is needed to unpack the RAR archive."
  if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
    apt-get update -qq && apt-get install -y -qq libarchive-tools
  else
    echo "Install it first, e.g.  sudo apt-get install libarchive-tools"; exit 1
  fi
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "Downloading LungHist700 (~670 MB)..."
curl -fL --retry 3 -o "$TMP/LungHist700.rar" "$URL"
[ "$(wc -c < "$TMP/LungHist700.rar" | tr -d ' ')" = 670356405 ] || { echo "unexpected archive size"; exit 1; }
bsdtar -xf "$TMP/LungHist700.rar" -C "$TMP"

mkdir -p "$ORIG" dataset/ADC dataset/SCC
cp "$TMP/data/data.csv" "$ORIG/data.csv"
for g in aca_bd aca_md aca_pd scc_bd scc_md scc_pd nor; do
  mkdir -p "$ORIG/$g"
  cp "$TMP/data/images/$g/"*.jpg "$ORIG/$g/"
done
for g in aca_bd aca_md aca_pd; do cp "$ORIG/$g/"*.jpg dataset/ADC/; done
for g in scc_bd scc_md scc_pd; do cp "$ORIG/$g/"*.jpg dataset/SCC/; done

[ "$(count dataset/ADC)" = 280 ] && [ "$(count dataset/SCC)" = 260 ] \
  || { echo "unexpected counts: ADC $(count dataset/ADC), SCC $(count dataset/SCC)"; exit 1; }
echo "LungHist700 ready: 280 ADC, 260 SCC, $(($(wc -l < "$ORIG/data.csv") - 1)) rows in data.csv."
