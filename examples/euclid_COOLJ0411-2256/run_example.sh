#!/usr/bin/env bash
# Euclid example: photometry of the COOLJ0411-2256 giant arc in VIS, Y, J and H, then one merged catalogue.
# Run from this folder:  bash run_example.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/../.."
IMG="$HERE/data/COOLJ0411-2256_euclid_dr1_stamp.fits"     # 30"x30" stamp of the Euclid DR1 cutout
REG="$HERE/COOLJ0411-2256_arc.reg"
OUT="$HERE/output"

# One run per band: -e picks the extension by name, --prefix labels the outputs.
# Euclid is detected from the MAGZERO keyword (or force it with -t EUCLID).
for pair in VIS:VIS NIR-Y:Y NIR-J:J NIR-H:H; do
    ext="${pair%%:*}"; label="${pair##*:}"
    python "$CODE/custom_aperture_photometry.py" -i "$IMG" -e "$ext" -r "$REG" --prefix "$label" -o "$OUT"
done

# Merge the four per-band catalogues on region_name and add colours.
cd "$OUT"
python "$CODE/merge_photometry.py" VIS_custom_photometry_results.fits Y_custom_photometry_results.fits \
    J_custom_photometry_results.fits H_custom_photometry_results.fits \
    --color VIS-H --color Y-H --color J-H \
    --output COOLJ0411-2256_euclid_merged.fits --output-txt COOLJ0411-2256_euclid_merged.txt
