# Custom aperture photometry from DS9 regions

This folder contains the aperture photometry code used for SLICE and LEGGOS. You draw apertures of any shape in DS9 (polygons, circles or ellipses). The code measures the flux in each aperture and subtracts a local background that you also draw. It returns fluxes, AB magnitudes and a breakdown of the errors. A second script merges the per-band catalogues into one table with colours.

The code was written by **Guillaume Mahler** (versions 1–11). Version 12 adds Euclid support and a few fixes, listed under [Changes in v12](#changes-in-v12). If you use it, please credit Guillaume.

| File | What it does |
|---|---|
| `custom_aperture_photometry.py` | Photometry of one image (one band) with one DS9 region file. Supports JWST, HST, DECaLS/DECam and Euclid. |
| `merge_photometry.py` | Joins several per-band catalogues on the region name and adds colours. |
| `examples/euclid_COOLJ0411-2256/` | A worked example on a Euclid DR1 cutout: the region file, `run_example.sh` and its outputs. |

**Requirements:** Python 3 with `numpy`, `astropy` and `matplotlib`.

## 1. Draw the regions in DS9

Draw the regions on the image and save them in **fk5** (or image) coordinates. Give every region a text label, because the label decides what the region is used for:

| Label | Used for |
|---|---|
| `rms`, `rms1`, `rms2`, … | **Required, at least one.** Empty sky. The sigma-clipped standard deviation of these pixels is the per-pixel noise, `std_rms`. With several regions the median is used. |
| `bck_<name>` | The local background for source `<name>`. The code uses the pixels inside `bck_<name>` but outside `<name>`, so you can draw it as a slightly larger copy of the source shape. The background level is the sigma-clipped median of those pixels. |
| anything else | A source to measure. |

Notes:

- Write circle and ellipse radii in **arcseconds** (`1.5"`), which is how DS9 saves fk5 regions by default. Arcminute and degree units are not parsed.
- A pixel belongs to a polygon if its centre is inside it. There is no partial-pixel weighting.
- Sources without a `bck_` region get no local background subtraction, and the code prints a warning. The image is never background-subtracted as a whole.

## 2. Run the photometry, one band at a time

```bash
python custom_aperture_photometry.py -i IMAGE.fits -r REGIONS.reg --prefix LABEL [-e EXT] [-o OUTDIR]
```

The telescope is read from the header keyword `TELESCOP`, or chosen with `-t`.

| Telescope | Input units | How the flux is calibrated |
|---|---|---|
| `JWST` | MJy/sr (`i2d`), nJy, µJy or Jy | converted to µJy using the pixel area; an `ERR` extension is used if present |
| `HST` | electrons/s | AB zeropoint from `PHOTFLAM` and `PHOTPLAM` |
| `DECALS` / `DECAM` | nanomaggies | `FLUXZP` (22.5 if missing); divided by `MWTRANS` for Milky Way dust; inverse-variance map via `--ivar` |
| `EUCLID` | ADU/s (VIS), e⁻/s (NISP) | AB zeropoint `MAGZERO` from the header, or `--zeropoint`; chosen automatically when the header has `MAGZERO` but no `TELESCOP` |

Useful options:

| Option | Effect |
|---|---|
| `-e`, `--extension` | Science extension, by index or name (`-e 1`, `-e SCI`, `-e VIS`, `-e NIR-H`). |
| `--prefix` | Label added to the start of every output file name. Use the filter name, because the merge step reads the label from the file name. |
| `-o`, `--outdir` | Where to write the outputs (default: the current directory). |
| `--err-extension`, `--no-err-map` | Choose or switch off the per-pixel error map (default `ERR`). |
| `--cal-uncertainty` | Absolute-calibration error as a fraction of the flux. JWST defaults to 2% for F150W2 and 4% for F322W2; Euclid defaults to 0. |
| `-v` | Print detailed diagnostics for each source. |

### Outputs (each name starts with `<prefix>_`)

| File | Content |
|---|---|
| `custom_photometry_results.fits` / `.txt` | One row per source: `region_name`, `ra_deg` and `dec_deg` (the centroid of the aperture), `n_pixels`, flux in µJy and mJy (plus nJy and nanomaggies for DECaLS), `magnitude` and `mag_error`, and each error term separately. |
| `photometry_<name>.png` | The cutout, the cutout with the local background subtracted (the annulus drawn in green), and the aperture mask. |
| `mask_<name>.fits`, `annulus_masks.fits` | The aperture and annulus masks, for checking in DS9. |
| `custom_photometry_overview.png` | The whole image with every source labelled. |

A non-positive flux is written as magnitude `99.999`.

### Error budget

The flux error combines these terms in quadrature:

1. **Sky noise in the aperture:** `std_rms × √N_pix`.
2. **Uncertainty in the subtracted background:** `std_rms × N_pix / √N_annulus` (Bevington & Robinson). Only applied when the source has a `bck_` region.
3. **Error map:** the square root of the sum of the `ERR` pixels squared (JWST/HST), or of `1/ivar` (DECaLS).
4. **Absolute calibration:** `cal_frac × |flux|` (JWST, or any telescope with `--cal-uncertainty`).

## 3. Merge the bands

```bash
python merge_photometry.py F150W2_custom_photometry_results.fits F322W2_custom_photometry_results.fits \
    --color F150W2-F322W2 --output merged.fits --output-txt merged.txt
```

- The band label is read from the file name `<LABEL>_custom_photometry_results.fits`. Use `--labels` to set it yourself.
- The tables are joined on `region_name`. By default this is an outer join, so a source missing from one band is kept and that band's values are masked.
- RA, Dec and `n_pixels` come from the **first** catalogue.
- Each band keeps `flux_microJy`, `flux_microJy_error`, `magnitude` and `mag_error`.
- Each `--color A-B` adds `color_A_minus_B` and its error.

## Example: Euclid DR1, COOLJ0411-2256

`examples/euclid_COOLJ0411-2256/` measures the giant arc in the four Euclid bands, then merges them. It runs on a 30″ × 30″ stamp included in the folder (`data/COOLJ0411-2256_euclid_dr1_stamp.fits`, 1.7 MB). The stamp was cut from the Euclid DR1 MER mosaics on IRSA and has the same layout as the COOL-LAMPS cutouts: extensions `VIS`, `NIR-Y`, `NIR-J`, `NIR-H` and `VIS_FLAG`, each with `MAGZERO`.

```bash
cd examples/euclid_COOLJ0411-2256 && bash run_example.sh
```

The region file has one polygon along the arc (`arc`), its background region (`bck_arc`) and one patch of empty sky (`rms`). The results are in `output/`. The full-size mask FITS files were left out to keep the folder small; running `run_example.sh` recreates them.

| Band | AB mag |
|---|---|
| VIS | 22.74 ± 0.01 |
| Y | 23.22 ± 0.03 |
| J | 22.67 ± 0.02 |
| H | 22.30 ± 0.02 |

**This example shows how to run the code. It is not a measurement of the arc.** The aperture is 0.7″ wide and the bands are **not PSF-matched**: VIS has a 0.16″ PSF and NISP about 0.5″, so the NISP bands lose more light outside the aperture. The background annulus also lies on the halo of the central galaxy. In the NISP bands, especially Y, it picks up halo and arc light, which over-subtracts the arc. For science colours:

- PSF-match the images first.
- Make the aperture wide compared with the broadest PSF.
- Put the `bck_` region on clean sky.

## Changes in v12

All changes are marked `v12` in the code.

- **New Euclid branch** (`perform_euclid_photometry`). Euclid MER pixels are linear count rates with an AB zeropoint `MAGZERO`. In DR1 the zeropoints are VIS 24.5 (ADU/s) and Y, J, H 29.8, 30.1, 30.0 (e⁻/s). Pixels are converted to µJy with the factor `10**(-0.4*(MAGZERO-23.9))` and then measured with the JWST routine. Every step of that routine is linear in the pixel values, so the result is the same as working in native units.
- **Extensions by name.** `-e` accepts an extension name as well as an index.
- **Output folder.** `-o/--outdir` sets where the outputs are written.
- **Bug fix.** In JWST and DECaLS rows of v11, the `flux_milliJy*` columns were 1000 times too small: µJy had been multiplied by 1e-6 instead of 1e-3. The µJy, nJy, nanomaggy and magnitude columns, which are the ones `merge_photometry.py` uses, were always correct.
- **Bug fix.** v11 read region files saved in DS9 **image** coordinates as if they counted pixels from 0. DS9 counts from 1, so each of those apertures sat one pixel up and to the right of where it was drawn. On a synthetic test image this lost 19% of the flux of a 10 × 10-pixel box. Region files in fk5 or icrs coordinates were measured correctly.
- **Unchanged.** `merge_photometry.py` is unchanged from Guillaume's version.

## Caveats

- **No PSF matching.** Each band is measured on its own image as given. To compare bands, PSF-match the images first.
- **Ellipses.** The DS9 ellipse angle is converted with `angle + 90°` (from v11). Polygons and circles have been used most; check ellipse masks in the `photometry_<name>.png` plots before trusting them.
- **NaN pixels.** NaNs in the science image are set to 0 and add no flux. Check that apertures don't run off the edge of the data.
