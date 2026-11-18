#!/usr/bin/env python3
"""
Generalized merger for N custom-aperture photometry catalogs.

Usage:
    python merge_photometry.py cat1.fits cat2.fits cat3.fits ...

By default, filter labels are extracted from the basename of each input:
    "{LABEL}_custom_photometry_results.fits"  ->  LABEL
So `F150W2_custom_photometry_results.fits` becomes label `F150W2`. Override
with --labels FILTER1 FILTER2 ... (one per catalog, in the same order).

All non-key columns are tagged with their label (e.g. `magnitude_F150W2`)
and the tables are joined sequentially on `region_name`.

ASTROMETRY: RA/Dec for the merged catalog are taken from the FIRST catalog
passed on the command line. All per-filter ra_deg_* / dec_deg_* columns
are then dropped.

COLORS: by default none are computed. Use --color LABEL1-LABEL2 (repeatable)
to add specific color columns, e.g. --color F150W2-F322W2.
"""

import argparse
import os
import sys
import numpy as np
from astropy.table import Table, join


# ======== Canonical column set for the slimmed output ========
# Per-filter, REQUIRED: kept always; warned if missing (shouldn't happen).
PER_FILTER_REQUIRED = [
    "flux_microJy",
    "flux_microJy_error",
    "magnitude",
    "mag_error",
]
# Per-filter, OPTIONAL: kept if present in the input catalog; warned (once
# per filter) and skipped if absent. This is what handles the nanomaggy
# columns -- they will be present once the photometry is rerun with the
# updated pipeline that emits them, and silently skipped otherwise.
PER_FILTER_OPTIONAL = [
    "flux_nanomaggy",
    "flux_nanomaggy_error",
]
# =============================================================


def extract_filter_name(path):
    """Extract the filter label from '{LABEL}_custom_photometry_results.fits'."""
    base = os.path.basename(path)
    base_no_ext = os.path.splitext(base)[0]
    suffix = "_custom_photometry_results"
    if base_no_ext.endswith(suffix):
        return base_no_ext[: -len(suffix)]
    return base_no_ext  # fall back: use whole basename


def tag_columns(t, tag, key):
    """Rename every column except `key` by appending `_{tag}` to it."""
    for col in list(t.colnames):
        if col != key:
            t.rename_column(col, f"{col}_{tag}")
    return t


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("catalogs", nargs="+",
                   help="Photometry FITS catalogs to merge. Order matters: "
                        "the FIRST catalog supplies the RA/Dec used in the output.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Override the auto-detected filter labels (one per catalog, "
                        "same order). Default: extracted from basename.")
    p.add_argument("--output", default="merged_photometry.fits",
                   help="Output FITS filename (default: merged_photometry.fits)")
    p.add_argument("--output-txt", default="merged_photometry.txt",
                   help="Output ASCII filename (default: merged_photometry.txt). "
                        "Use 'none' to skip.")
    p.add_argument("--match-key", default="region_name",
                   help="Column to join on (default: region_name).")
    p.add_argument("--join-type", default="outer", choices=["inner", "outer"],
                   help="'outer' keeps regions present in ANY catalog (masked where "
                        "missing); 'inner' keeps only regions present in ALL. "
                        "Default: outer.")
    p.add_argument("--color", action="append", default=[],
                   metavar="LABEL1-LABEL2",
                   help="Compute a color column 'LABEL1 - LABEL2' (in magnitudes). "
                        "Repeatable. Example: --color F150W2-F322W2 --color DECaLSg-DECaLSi.")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Resolve labels ---
    if args.labels is not None:
        if len(args.labels) != len(args.catalogs):
            sys.exit(f"ERROR: --labels has {len(args.labels)} entries but "
                     f"{len(args.catalogs)} catalogs were given.")
        labels = args.labels
    else:
        labels = [extract_filter_name(p) for p in args.catalogs]

    # Refuse duplicate labels (would cause column-rename collisions)
    if len(set(labels)) != len(labels):
        seen, dups = set(), []
        for x in labels:
            if x in seen and x not in dups:
                dups.append(x)
            seen.add(x)
        sys.exit(f"ERROR: duplicate labels {dups}; use --labels to disambiguate.")

    print(f"Will merge {len(args.catalogs)} catalogs.")
    print(f"RA/Dec will be inherited from the first catalog: '{labels[0]}'.")

    # --- Read and tag each table ---
    tables = []
    for path, label in zip(args.catalogs, labels):
        t = Table.read(path)
        print(f"  Loaded {len(t):>3d} rows from {path}  ->  label '{label}'")
        if "prefix" in t.colnames:
            t.remove_column("prefix")
        t = tag_columns(t, label, args.match_key)
        tables.append(t)

    # --- Sequential join ---
    merged = tables[0]
    for t in tables[1:]:
        merged = join(merged, t, keys=args.match_key, join_type=args.join_type)
    print(f"Merged: {len(merged)} rows, {len(merged.colnames)} columns "
          f"(join_type='{args.join_type}').")

    # Report unmatched regions per catalog (only meaningful for outer join)
    if args.join_type == "outer":
        all_keys = set(merged[args.match_key])
        for path, label, t in zip(args.catalogs, labels, tables):
            keys_here = set(t[args.match_key])
            missing = all_keys - keys_here
            if missing:
                print(f"  '{label}' is missing {len(missing)} region(s) "
                      f"(present in other catalogs).")

    # --- RA/Dec: take from the first catalog, drop all per-filter copies ---
    first_label = labels[0]
    ra_first  = f"ra_deg_{first_label}"
    dec_first = f"dec_deg_{first_label}"
    if ra_first in merged.colnames and dec_first in merged.colnames:
        merged["ra_deg"]  = merged[ra_first]
        merged["dec_deg"] = merged[dec_first]
        to_drop = [c for c in merged.colnames
                   if c.startswith("ra_deg_") or c.startswith("dec_deg_")]
        merged.remove_columns(to_drop)
        print(f"  RA/Dec inherited from '{first_label}'; per-filter ra/dec columns dropped.")
    else:
        print(f"  WARNING: '{ra_first}' / '{dec_first}' not found in the first "
              f"catalog -- skipping RA/Dec consolidation. (Re-run the photometry "
              f"with v7 of custom_aperture_photometry to include ra_deg/dec_deg.)")

    # --- Optional colors ---
    for spec in args.color:
        if "-" not in spec:
            print(f"  WARNING: invalid --color '{spec}' (expected LABEL1-LABEL2). Skipped.")
            continue
        a, b = spec.split("-", 1)
        m_a, m_b = f"magnitude_{a}",   f"magnitude_{b}"
        e_a, e_b = f"mag_error_{a}",   f"mag_error_{b}"
        if m_a not in merged.colnames or m_b not in merged.colnames:
            print(f"  WARNING: missing magnitude column for color '{spec}'. Skipped.")
            continue
        col_name = f"color_{a}_minus_{b}"
        merged[col_name]              = merged[m_a] - merged[m_b]
        merged[f"{col_name}_error"]   = np.sqrt(merged[e_a] ** 2 + merged[e_b] ** 2)
        print(f"  Added color column: {col_name}")

    # --- Slim the table to the canonical column set ---
    # Shared columns first, then per-filter blocks in catalog order, then colors.
    keep = [args.match_key]
    if "ra_deg"  in merged.colnames: keep.append("ra_deg")
    if "dec_deg" in merged.colnames: keep.append("dec_deg")

    # n_pixels: keep ONLY the first catalog's, with its label suffix retained
    # (so the meaning -- which filter's pixel scale -- stays explicit).
    first_npix = f"n_pixels_{first_label}"
    if first_npix in merged.colnames:
        keep.append(first_npix)
    else:
        print(f"  WARNING: '{first_npix}' not in merged table -- n_pixels omitted.")

    # Per-filter flux/mag block
    for lbl in labels:
        # Check optional columns once and warn together
        missing_opt = [b for b in PER_FILTER_OPTIONAL
                       if f"{b}_{lbl}" not in merged.colnames]
        if missing_opt:
            print(f"  WARNING: optional columns missing from '{lbl}' catalog "
                  f"-- skipped: {missing_opt}")

        for base in PER_FILTER_REQUIRED + PER_FILTER_OPTIONAL:
            col = f"{base}_{lbl}"
            if col in merged.colnames:
                keep.append(col)
            elif base in PER_FILTER_REQUIRED:
                print(f"  WARNING: required column '{col}' missing -- skipped.")

    # Color columns (computed just above, always present if --color was used)
    for spec in args.color:
        if "-" in spec:
            a, b = spec.split("-", 1)
            cn = f"color_{a}_minus_{b}"
            if cn in merged.colnames:
                keep.append(cn)
                keep.append(f"{cn}_error")

    n_dropped = len(merged.colnames) - len(keep)
    merged = merged[keep]
    print(f"  Slimmed to {len(keep)} columns ({n_dropped} dropped).")

    # --- Save ---
    merged.write(args.output, overwrite=True)
    print(f"Wrote {args.output}  ({len(merged)} rows, {len(merged.colnames)} cols)")

    if args.output_txt and args.output_txt.lower() != "none":
        merged.write(args.output_txt, format="ascii.commented_header", overwrite=True)
        print(f"Wrote {args.output_txt}")

    # --- Quick inspection: key, ra/dec, all magnitudes, all colors ---
    summary_cols = [args.match_key]
    for c in ("ra_deg", "dec_deg"):
        if c in merged.colnames:
            summary_cols.append(c)
    for lbl in labels:
        for c in (f"magnitude_{lbl}", f"mag_error_{lbl}"):
            if c in merged.colnames:
                summary_cols.append(c)
    for spec in args.color:
        if "-" in spec:
            a, b = spec.split("-", 1)
            cn = f"color_{a}_minus_{b}"
            if cn in merged.colnames:
                summary_cols.append(cn)
                summary_cols.append(f"{cn}_error")
    print()
    print(merged[summary_cols])


if __name__ == "__main__":
    main()
