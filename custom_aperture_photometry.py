#!/usr/bin/env python
"""
Custom Aperture Photometry from DS9 Region Files (v12)
Performs photometry on user-defined apertures with flexible unit conversion.

Original code (v1-v11): Guillaume Mahler (SLICE collaboration).

v12: Euclid support and packaging fixes
---------------------------------------
  - New 'EUCLID' telescope branch (perform_euclid_photometry): Euclid MER mosaics / cutouts are linear
    count rates with an AB zeropoint MAGZERO in the header (VIS 24.5 ADU/s; NISP Y/J/H 29.8/30.1/30.0
    e-/s in DR1). Pixels are converted to µJy with 10**(-0.4*(MAGZERO-23.9)) and measured exactly as
    for JWST. Auto-selected when the header has MAGZERO but no TELESCOP; --zeropoint overrides MAGZERO.
  - -e/--extension accepts an extension NAME as well as an index (e.g. -e VIS, -e NIR-H).
  - -o/--outdir writes all outputs to a chosen directory (default: the current directory).
  - Bug fix: the milliJy columns of JWST and DECaLS rows were 1000x too small (µJy were multiplied by
    1e-6 instead of 1e-3). The microJy, nJy, nanomaggy and magnitude columns were always correct.
  - Bug fix: region files in DS9 'image' coordinates (1-indexed) were used as 0-indexed pixel positions,
    shifting those apertures by +1 pixel in x and y. Sky-coordinate (fk5/icrs) region files were not affected.

v10: local annular background subtraction
-----------------------------------------
The Background2D-based machinery of v8/v9 has been removed entirely.
Background is now estimated LOCALLY for each source from a user-drawn
companion region. The DS9 region file is parsed into three categories
based on the region's text label:

  - 'rms', 'rms1', 'rms2', ... : RMS noise regions (unchanged from v9).
  - 'bck_<name>'               : local background region for source <name>.
                                 The annulus used for the background level
                                 is (bck_mask AND NOT source_mask), so the
                                 source pixels are excluded even if the bck
                                 polygon overlaps them.
  - everything else            : photometry sources.

For each source 'xxx' that has a matching 'bck_xxx' region, the local
background per pixel is the sigma-clipped median over the annulus. The
flux total is corrected as:

    flux_corrected = sum(data[source_mask]) - local_bkg * n_source_pixels

and an additional error term std_rms * n_pixels / sqrt(n_annulus) is added
in quadrature to account for the uncertainty in the subtracted level
(Bevington & Robinson form). Sources without a matching bck_ region get
no local subtraction and the corresponding error term is zero; a warning
is printed.

No global background subtraction is performed; the science image is used
as-is. The annulus masks are saved as a labeled FITS image for QA in DS9.

v11: DECaLS / DECam support
----------------------------
A new telescope branch ('DECALS' / 'DECAM') handles DECaLS Legacy Survey
single-band images whose pixel values are in nanomaggies (linear flux).

Key additions:
  - perform_decals_photometry(): dedicated photometry function for DECaLS.
      * Reads FLUXZP from header (falls back to 22.5 if absent).
      * Reads MWTRANS and divides flux by it for Milky Way dust correction
        (warns if the keyword is absent; correction is skipped, not fatal).
      * Sums pixels linearly (nanomaggies are per-pixel flux, not surface
        brightness; summing is correct for total flux in the aperture).
      * Outputs flux in nanomaggies, AB magnitude, and nanoJansky.
      * Error budget (combined in quadrature):
          (1) RMS-region polygon term: std_rms * sqrt(N_pix)
          (2) Bevington local-background term: std_rms * N_pix / sqrt(N_ann)
          (3) ivar-map term (preferred): sqrt(sum(1/ivar)) over valid pixels,
              where pixels with ivar=0 are excluded (warned once per source).
              Falls back to zero if no ivar map is supplied.
  - load_fits_image(): 'ERR' extension logic extended; for DECaLS the caller
    passes err_ext=None and supplies an ivar map via a separate path.
  - custom_aperture_photometry(): new ivar_name parameter; the ivar array is
    loaded and forwarded to perform_decals_photometry() as ivar_data.
  - CLI: --telescope DECALS/DECAM option added; --ivar argument for the
    inverse-variance map FITS file (optional).
  - save_results(): result table extended with flux_nanomaggy, flux_nJy
    columns (populated for DECaLS, NaN for JWST/HST rows).
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.path import Path
import re
import sys
import argparse

def load_fits_image(filename, science_ext=0, err_ext='ERR'):
    """
    Load FITS image and extract data, header, WCS, and optionally an
    ERR (1-sigma uncertainty) map.

    Parameters:
    -----------
    filename : str
        Path to FITS file
    science_ext : int
        FITS extension for science data (0 for primary, 1 for first extension, etc.)
    err_ext : int, str, or None
        FITS extension (by index or name) containing the per-pixel 1-sigma
        error map. JWST i2d products contain a 2D HDU named 'ERR'. Pass
        None to skip ERR-map loading (e.g. for HST drz/drc files that have
        no ERR extension).

    Returns:
    --------
    data : numpy array
        Image data with NaNs replaced by zeros
    header : FITS header
        Image header
    wcs : WCS object
        World Coordinate System information
    err_data : numpy array or None
        Per-pixel 1-sigma error map (same shape/units as `data`) if found,
        otherwise None. NaNs are replaced by zeros (no error contribution).
    """
    print(f"1. Loading FITS image...")

    err_data = None
    try:
        with fits.open(filename) as hdul:
            print(f"   Loading science data from extension {science_ext}...")
            data = hdul[science_ext].data.copy()
            header = hdul[science_ext].header
            print(f"   Image shape: {data.shape}")

            # Get WCS for coordinate conversion
            wcs = WCS(header)
            print(f"   WCS loaded successfully")

            # Optionally load the ERR (1-sigma uncertainty) extension.
            if err_ext is not None:
                try:
                    err_hdu = hdul[err_ext]
                    err_data = err_hdu.data.copy()
                    if err_data.shape != data.shape:
                        print(f"   WARNING: ERR extension shape {err_data.shape} "
                              f"does not match SCI shape {data.shape}. "
                              f"Skipping ERR-map contribution.")
                        err_data = None
                    else:
                        print(f"   Loaded ERR extension '{err_ext}' "
                              f"(shape {err_data.shape})")
                except (KeyError, IndexError):
                    print(f"   No ERR extension '{err_ext}' found "
                          f"-- continuing without ERR-map error contribution.")
                    err_data = None

    except Exception as e:
        print(f"ERROR loading FITS file: {e}")
        sys.exit(1)

    # Handle NaN values in SCI
    nan_mask = np.isnan(data)
    n_nans = np.sum(nan_mask)
    print(f"   NaN pixels (SCI): {n_nans}")
    if n_nans > 0:
        data[nan_mask] = 0.0
        print("   NaN values in SCI replaced with zeros")

    # Handle NaN values in ERR (treat as zero uncertainty, i.e. no contribution)
    if err_data is not None:
        err_nan_mask = np.isnan(err_data)
        n_err_nans = int(np.sum(err_nan_mask))
        print(f"   NaN pixels (ERR): {n_err_nans}")
        if n_err_nans > 0:
            err_data[err_nan_mask] = 0.0
            print("   NaN values in ERR replaced with zeros")

    return data, header, wcs, err_data


def calculate_pixel_scale(header, wcs):
    """
    Calculate pixel scale and pixel area from header/WCS.
    
    Parameters:
    -----------
    header : FITS header
        Image header
    wcs : WCS object
        World Coordinate System information
    
    Returns:
    --------
    pixel_scale_arcsec : float
        Pixel scale in arcseconds per pixel
    pixel_area_sr : float
        Pixel area in steradians
    """
    print("\n   Calculating pixel scale...")
    
    # Try CDELT keywords first
    if 'CDELT1' in header and 'CDELT2' in header:
        cdelt1 = header['CDELT1']
        cdelt2 = header['CDELT2']
        
        # Check if CDELT values seem to be in degrees
        if abs(cdelt1) < 0.001 and abs(cdelt2) < 0.001:
            pixel_scale_deg = np.sqrt(abs(cdelt1 * cdelt2))
            pixel_scale_arcsec = pixel_scale_deg * 3600
            pixel_area_sr = abs(cdelt1 * cdelt2) * (np.pi / 180)**2
            print(f"   Using CDELT keywords: CDELT1={cdelt1:.6e}°, CDELT2={cdelt2:.6e}°")
            print(f"   Pixel scale: {pixel_scale_arcsec:.4f} arcsec/pixel")
            print(f"   Pixel area: {pixel_area_sr:.6e} sr/pixel")
        else:
            # CDELT might be 1.0 for reprojected images
            print(f"   CDELT values ({cdelt1}, {cdelt2}) suggest reprojected image")
            print(f"   Using WCS to determine pixel scale...")
            try:
                pix_00 = wcs.pixel_to_world(0, 0)
                pix_10 = wcs.pixel_to_world(1, 0)
                pix_01 = wcs.pixel_to_world(0, 1)
                
                sep_x = pix_00.separation(pix_10).to(u.arcsec).value
                sep_y = pix_00.separation(pix_01).to(u.arcsec).value
                
                pixel_scale_arcsec = np.sqrt(sep_x * sep_y)
                pixel_scale_deg = pixel_scale_arcsec / 3600
                pixel_area_sr = (sep_x / 3600) * (sep_y / 3600) * (np.pi / 180)**2
                
                print(f"   Pixel scale from WCS: {pixel_scale_arcsec:.4f} arcsec/pixel")
                print(f"   Pixel area: {pixel_area_sr:.6e} sr/pixel")
            except:
                pixel_scales = wcs.proj_plane_pixel_scales()
                pixel_scale_deg = np.mean([abs(ps.to(u.deg).value) for ps in pixel_scales])
                pixel_scale_arcsec = pixel_scale_deg * 3600
                pixel_area_sr = pixel_scale_deg**2 * (np.pi / 180)**2
                print(f"   Using WCS pixel scales (fallback)")
                print(f"   Pixel scale: {pixel_scale_arcsec:.4f} arcsec/pixel")
                print(f"   Pixel area: {pixel_area_sr:.6e} sr/pixel")
    
    # Fallback to CD matrix or WCS
    elif hasattr(wcs, 'pixel_scale_matrix'):
        cd_matrix = wcs.pixel_scale_matrix
        pixel_scale_deg = np.sqrt(abs(cd_matrix[0, 0] * cd_matrix[1, 1] - cd_matrix[0, 1] * cd_matrix[1, 0]))
        pixel_scale_arcsec = pixel_scale_deg * 3600
        pixel_area_sr = pixel_scale_deg**2 * (np.pi / 180)**2
        print(f"   Using CD matrix (CDELT not found)")
        print(f"   Pixel scale: {pixel_scale_arcsec:.4f} arcsec/pixel")
        print(f"   Pixel area: {pixel_area_sr:.6e} sr/pixel")
    else:
        try:
            pixel_scales = wcs.proj_plane_pixel_scales()
            pixel_scale_deg = np.mean([abs(ps.to(u.deg).value) for ps in pixel_scales])
            pixel_scale_arcsec = pixel_scale_deg * 3600
            pixel_area_sr = pixel_scale_deg**2 * (np.pi / 180)**2
            print(f"   Using WCS pixel scales")
            print(f"   Pixel scale: {pixel_scale_arcsec:.4f} arcsec/pixel")
            print(f"   Pixel area: {pixel_area_sr:.6e} sr/pixel")
        except:
            print("   WARNING: Could not determine pixel scale from header!")
            print("   Using approximate value for space telescopes")
            pixel_scale_arcsec = 0.063
            pixel_scale_deg = pixel_scale_arcsec / 3600
            pixel_area_sr = pixel_scale_deg**2 * (np.pi / 180)**2
    
    return pixel_scale_arcsec, pixel_area_sr


def parse_ds9_regions(region_file):
    """
    Parse DS9 region file and return list of regions.
    
    Parameters:
    -----------
    region_file : str
        Path to DS9 region file
    
    Returns:
    --------
    regions : list
        List of region dictionaries
    """
    print("\n2. Parsing DS9 region file...")
    regions = []
    coordinate_system = 'image'  # default
    
    with open(region_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check for coordinate system
            if line in ['fk5', 'fk4', 'icrs', 'galactic', 'ecliptic']:
                coordinate_system = line
                print(f"   Coordinate system: {coordinate_system}")
                continue
            elif line in ['image', 'physical']:
                coordinate_system = line
                continue
                
            # Parse region shapes
            if 'circle(' in line:
                match = re.search(r'circle\((.*?)\)', line)
                if match:
                    coords = match.group(1).split(',')
                    ra, dec, radius = float(coords[0]), float(coords[1]), coords[2].strip('"')
                    radius = float(radius.replace('"', ''))
                    
                    text_match = re.search(r'text=\{(.*?)\}', line)
                    name = text_match.group(1) if text_match else f"circle_{len(regions)}"
                    
                    regions.append({
                        'type': 'circle',
                        'coords': (ra, dec, radius),
                        'name': name,
                        'coord_system': coordinate_system
                    })
                    
            elif 'ellipse(' in line:
                match = re.search(r'ellipse\((.*?)\)', line)
                if match:
                    coords = match.group(1).split(',')
                    ra, dec = float(coords[0]), float(coords[1])
                    semi_major = float(coords[2].strip('"').replace('"', ''))
                    semi_minor = float(coords[3].strip('"').replace('"', ''))
                    angle = float(coords[4]) if len(coords) > 4 else 0.0
                    
                    text_match = re.search(r'text=\{(.*?)\}', line)
                    name = text_match.group(1) if text_match else f"ellipse_{len(regions)}"
                    
                    regions.append({
                        'type': 'ellipse',
                        'coords': (ra, dec, semi_major, semi_minor, angle),
                        'name': name,
                        'coord_system': coordinate_system
                    })
                    
            elif 'polygon(' in line:
                match = re.search(r'polygon\((.*?)\)', line)
                if match:
                    coords_str = match.group(1)
                    coords = [float(x) for x in coords_str.split(',')]
                    coord_pairs = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    
                    text_match = re.search(r'text=\{(.*?)\}', line)
                    name = text_match.group(1) if text_match else f"polygon_{len(regions)}"
                    
                    regions.append({
                        'type': 'polygon',
                        'coords': coord_pairs,
                        'name': name,
                        'coord_system': coordinate_system
                    })
    
    print(f"   Found {len(regions)} regions:")
    for region in regions:
        print(f"     - {region['name']}: {region['type']}")
    
    return regions


def categorize_regions(regions):
    """
    Categorize DS9 regions into three groups by their text label.

      - RMS regions   : name matches '^rms\\d*$' (case-insensitive).
                        Used for the per-pixel noise estimate (std_rms).
      - bck regions   : name matches '^bck_(.+)$' (case-insensitive).
                        Provide a local background polygon for source <name>.
                        The annulus used for the background is the bck mask
                        with the source mask removed.
      - photometry    : everything else.

    Parameters
    ----------
    regions : list
        List of region dicts (output of parse_ds9_regions).

    Returns
    -------
    rms_regions : list of dict
        RMS noise regions.
    bck_regions : list of dict
        Local-background regions. Each dict has an added 'target' field
        equal to the source name it provides background for.
    photometry_regions : list of dict
        Source regions to measure.
    """
    RMS_NAME_RE = re.compile(r'^rms\d*$',     re.IGNORECASE)
    BCK_NAME_RE = re.compile(r'^bck_(.+)$',   re.IGNORECASE)

    print("\n3. Categorizing regions (rms / bck / photometry)...")

    rms_regions, bck_regions, photometry_regions = [], [], []
    for region in regions:
        name = region['name']
        if RMS_NAME_RE.match(name):
            rms_regions.append(region)
            print(f"   ✓ RMS region        : {name} ({region['type']})")
        elif BCK_NAME_RE.match(name):
            bck_copy = dict(region)
            bck_copy['target'] = BCK_NAME_RE.match(name).group(1)
            bck_regions.append(bck_copy)
            print(f"   ✓ Local-bkg region  : {name}  -> source '{bck_copy['target']}'")
        else:
            photometry_regions.append(region)
            print(f"   ✓ Photometry region : {name} ({region['type']})")

    if not rms_regions:
        print("\n   ERROR: No RMS region found in the region file!")
        print("   Add at least one region named 'rms' (or 'rms1', 'rms2', ...) "
              "for noise estimation.")
        sys.exit(1)

    # Cross-check: warn about bck targets that don't have a matching source,
    # and sources that have no bck region (these will get no local subtraction).
    source_names = {r['name'] for r in photometry_regions}
    bck_targets  = {b['target'] for b in bck_regions}

    orphan_bcks  = bck_targets - source_names
    sources_without_bck = source_names - bck_targets
    if orphan_bcks:
        print(f"\n   WARNING: {len(orphan_bcks)} bck_ region(s) reference "
              f"non-existent source(s): {sorted(orphan_bcks)}")
    if sources_without_bck:
        print(f"\n   WARNING: {len(sources_without_bck)} photometry source(s) "
              f"have no matching bck_ region (no local subtraction will be "
              f"applied for them): {sorted(sources_without_bck)}")

    print(f"\n   Totals: {len(rms_regions)} rms, {len(bck_regions)} bck, "
          f"{len(photometry_regions)} photometry")
    return rms_regions, bck_regions, photometry_regions


def sky_to_pixel_region(region, wcs, pixel_scale_arcsec):
    """
    Convert sky coordinates to pixel coordinates.
    
    Parameters:
    -----------
    region : dict
        Region dictionary
    wcs : WCS object
        World Coordinate System
    pixel_scale_arcsec : float
        Pixel scale in arcsec/pixel
    
    Returns:
    --------
    dict : Region in pixel coordinates
    """
    if region['coord_system'] in ('image', 'physical'):
        # v12: DS9 image coordinates are 1-indexed (the first pixel centre is 1,1); the masks use
        # 0-indexed numpy pixel centres, so shift positions by -1 (sizes and angles unchanged).
        # v11 used them unshifted, which moved every image-coordinate aperture by one pixel in x and y.
        # ('physical' is treated like 'image', i.e. assumes no LTM/LTV transform, as in v11.)
        out = {'type': region['type'], 'name': region['name']}
        c = region['coords']
        if region['type'] == 'polygon':
            out['coords'] = [(x - 1.0, y - 1.0) for x, y in c]
        else:
            out['coords'] = (c[0] - 1.0, c[1] - 1.0) + tuple(c[2:])
        return out
    
    print(f"   Converting {region['name']} ({region['type']}) from sky to pixel coordinates:")
    
    if region['type'] == 'circle':
        ra, dec, radius = region['coords']
        print(f"     Sky: RA={ra:.6f}°, Dec={dec:.6f}°, radius={radius:.3f}\"")
        
        sky_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        x, y = wcs.world_to_pixel(sky_coord)
        radius_pix = radius / pixel_scale_arcsec
        
        print(f"     Pixel: x={x:.2f}, y={y:.2f}, radius={radius_pix:.2f} pixels")
        print(f"     Using pixel scale: {pixel_scale_arcsec:.4f} arcsec/pixel")
        
        return {
            'type': 'circle',
            'coords': (x, y, radius_pix),
            'name': region['name']
        }
        
    elif region['type'] == 'ellipse':
        ra, dec, semi_major, semi_minor, angle = region['coords']
        print(f"     Sky: RA={ra:.6f}°, Dec={dec:.6f}°, semi_major={semi_major:.3f}\", semi_minor={semi_minor:.3f}\", angle={angle:.1f}°")
        
        sky_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        x, y = wcs.world_to_pixel(sky_coord)
        semi_major_pix = semi_major / pixel_scale_arcsec
        semi_minor_pix = semi_minor / pixel_scale_arcsec
        
        print(f"     Pixel: x={x:.2f}, y={y:.2f}, semi_major={semi_major_pix:.2f} pixels, semi_minor={semi_minor_pix:.2f} pixels, angle={angle:.1f}°")
        
        return {
            'type': 'ellipse',
            'coords': (x, y, semi_major_pix, semi_minor_pix, angle),
            'name': region['name']
        }
        
    elif region['type'] == 'polygon':
        coord_pairs = region['coords']
        print(f"     Sky coordinates: {len(coord_pairs)} vertices")
        for i, (ra, dec) in enumerate(coord_pairs):
            print(f"       Vertex {i+1}: RA={ra:.6f}°, Dec={dec:.6f}°")
            
        sky_coords = SkyCoord(ra=[coord[0] for coord in coord_pairs]*u.deg, 
                             dec=[coord[1] for coord in coord_pairs]*u.deg)
        pixel_coords = wcs.world_to_pixel(sky_coords)
        pixel_pairs = list(zip(pixel_coords[0], pixel_coords[1]))
        
        print(f"     Pixel coordinates:")
        for i, (x, y) in enumerate(pixel_pairs):
            print(f"       Vertex {i+1}: x={x:.2f}, y={y:.2f}")
        
        return {
            'type': 'polygon',
            'coords': pixel_pairs,
            'name': region['name']
        }


def create_region_mask(region, image_shape):
    """
    Create a boolean mask for a given region.
    
    Parameters:
    -----------
    region : dict
        Region dictionary with pixel coordinates
    image_shape : tuple
        Shape of the image
    
    Returns:
    --------
    mask : numpy array
        Boolean mask
    """
    mask = np.zeros(image_shape, dtype=bool)
    
    if region['type'] == 'circle':
        x_center, y_center, radius = region['coords']
        y_indices, x_indices = np.ogrid[:image_shape[0], :image_shape[1]]
        distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
        mask = distances <= radius
        
    elif region['type'] == 'ellipse':
        x_center, y_center, semi_major, semi_minor, angle = region['coords']
        y_indices, x_indices = np.ogrid[:image_shape[0], :image_shape[1]]
        
        # Convert DS9 angle convention
        angle_corrected = angle + 90.0
        angle_rad = np.radians(angle_corrected)
        cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
        
        x_rot = (x_indices - x_center) * cos_angle + (y_indices - y_center) * sin_angle
        y_rot = -(x_indices - x_center) * sin_angle + (y_indices - y_center) * cos_angle
        
        mask = (x_rot**2 / semi_major**2) + (y_rot**2 / semi_minor**2) <= 1
        
    elif region['type'] == 'polygon':
        coord_pairs = region['coords']
        path = Path(coord_pairs)
        
        x_coords, y_coords = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        mask_flat = path.contains_points(points)
        mask = mask_flat.reshape(image_shape)
    
    return mask


def convert_to_microjansky(data, bunit, pixel_area_sr=None):
    """
    Convert various flux units to microJansky.
    
    Parameters:
    -----------
    data : array
        Input flux data
    bunit : str
        Unit string from FITS header
    pixel_area_sr : float
        Pixel area in steradians (needed for surface brightness units)
    
    Returns:
    --------
    data_ujy : array
        Data converted to microJansky
    conversion_factor : float
        Factor used for conversion (for error propagation)
    """
    bunit_upper = bunit.upper().strip()
    
    if 'MJY/SR' in bunit_upper:
        if pixel_area_sr is None:
            raise ValueError("Pixel area needed for MJy/sr conversion")
        conversion_factor = pixel_area_sr * 1e12
        print(f"   Converting from MJy/sr to µJy using pixel area")
        print(f"   Conversion factor: {conversion_factor:.6e}")
        return data * conversion_factor, conversion_factor
    
    elif 'NANOJANSKY' in bunit_upper or 'NJY' in bunit_upper:
        scale = 1.0
        if '*' in bunit:
            try:
                scale_str = bunit.split('*')[0].strip()
                scale = float(scale_str)
                print(f"   Found scale factor: {scale}")
            except:
                scale = 1.0
        
        conversion_factor = scale * 1e-3
        print(f"   Converting from {scale}*nanoJansky to µJy")
        print(f"   Conversion factor: {conversion_factor:.6e}")
        return data * conversion_factor, conversion_factor
    
    elif 'MICROJANSKY' in bunit_upper or 'µJY' in bunit_upper or 'UJY' in bunit_upper:
        print(f"   Data already in microJansky")
        return data, 1.0
    
    elif 'JANSKY' in bunit_upper or 'JY' in bunit_upper:
        if 'NANO' not in bunit_upper and 'MICRO' not in bunit_upper and 'µ' not in bunit_upper:
            conversion_factor = 1e6
            print(f"   Converting from Jansky to µJy")
            print(f"   Conversion factor: {conversion_factor:.6e}")
            return data * conversion_factor, conversion_factor
    
    elif 'ELECTRONS/S' in bunit_upper or 'E/S' in bunit_upper:
        print(f"   WARNING: {bunit} requires instrument-specific calibration")
        print(f"   Please provide PHOTFLAM/PHOTFNU or use calibrated data")
        print(f"   Proceeding with data as-is (results will not be in physical units)")
        return data, 1.0
    
    else:
        print(f"   WARNING: Unknown unit '{bunit}'")
        print(f"   Proceeding without conversion (results may not be in microJansky)")
        return data, 1.0

def perform_hst_photometry(data, masks, bunit, pixel_area_sr, std_rms, header,
                           err_data=None, local_bkg=None, n_annulus=None,
                           verbose=True):
    """
    Perform HST photometry on masked regions using count rates and zeropoints.

    Parameters:
    -----------
    data : numpy array
        Raw image data in ELECTRONS/S (NOT pre-subtracted in v10).
    masks : dict
        Dictionary of masks for each region.
    bunit : str
        Unit string from FITS header (should be ELECTRONS/S for HST).
    pixel_area_sr : float
        Pixel area in steradians (not used for HST count rate photometry).
    std_rms : float
        RMS noise from RMS region (per-pixel, ELECTRONS/S).
    header : FITS header
        FITS header containing PHOTFLAM and PHOTPLAM.
    err_data : numpy array or None
        Optional per-pixel 1-sigma error map in ELECTRONS/S.
    local_bkg : dict {source_name: float} or None
        Per-source local background level (ELECTRONS/S per pixel). When
        provided for a source, `local_bkg * n_aperture_pixels` is
        subtracted from the count rate sum and the Bevington error term
        std_rms * n_pix / sqrt(n_annulus) is added in quadrature.
    n_annulus : dict {source_name: int} or None
        Per-source annulus pixel count, used for the propagated error.
    verbose : bool
        Print diagnostic information.

    Returns:
    --------
    results : list
        List of photometry results.
    """
    if local_bkg is None:
        local_bkg = {}
    if n_annulus is None:
        n_annulus = {}
    import astropy.units as u
    
    # Check if data is in the correct units
    if 'ELECTRONS/S' not in bunit.upper() and 'E/S' not in bunit.upper():
        print(f"   WARNING: Expected ELECTRONS/S units, got {bunit}")
        print(f"   Proceeding assuming count rate data...")
    
    # Extract PHOTFLAM and PHOTPLAM from header
    try:
        PHOTFLAM = header['PHOTFLAM']
        PHOTPLAM = header['PHOTPLAM']
        if verbose:
            print(f"   PHOTFLAM: {PHOTFLAM:.6e} erg cm^-2 s^-1 Å^-1 (e/s)^-1")
            print(f"   PHOTPLAM: {PHOTPLAM:.2f} Å")
    except KeyError as e:
        print(f"   ERROR: Missing HST calibration keyword: {e}")
        print(f"   Cannot perform HST photometry without PHOTFLAM and PHOTPLAM")
        return []
    
    # Calculate HST AB zeropoint
    zeropoint = compute_hst_zeropoint(PHOTFLAM, PHOTPLAM)
    if verbose:
        print(f"   HST AB zeropoint: {zeropoint:.3f} mag")
    
    results = []
    
    for region_name, mask in masks.items():
        region_data = data[mask]
        n_pixels = len(region_data)
        
        if n_pixels == 0:
            print(f"   WARNING: {region_name} has no pixels!")
            continue
        
        if verbose:
            print(f"\n   === HST Photometry for {region_name} ===")
            print(f"   Number of pixels: {n_pixels}")
            print(f"   Raw data statistics (ELECTRONS/S):")
            print(f"     Mean: {np.mean(region_data):.6e}")
            print(f"     Median: {np.median(region_data):.6e}")
            print(f"     Sum: {np.sum(region_data):.6e}")
            print(f"     Min: {np.min(region_data):.6e}")
            print(f"     Max: {np.max(region_data):.6e}")
        
        # Calculate count rate (sum of all pixels in ELECTRONS/S)
        count_rate_raw = np.sum(region_data)
        if verbose:
            print(f"   Count rate (raw):     {count_rate_raw:.6e} electrons/s")

        # --- v10: subtract local annular background, if available ---
        if region_name in local_bkg:
            lb_per_pix       = local_bkg[region_name]
            n_ann            = n_annulus.get(region_name, 0)
            local_bkg_native = lb_per_pix * n_pixels
            count_rate       = count_rate_raw - local_bkg_native
            if verbose:
                print(f"   Local annular background:")
                print(f"     per-pixel level : {lb_per_pix:.6e} electrons/s")
                print(f"     annulus pixels  : {n_ann}")
                print(f"     subtracted total: {local_bkg_native:.6e} electrons/s")
                print(f"   Count rate (corrected): {count_rate:.6e} electrons/s")
        else:
            count_rate = count_rate_raw
            n_ann      = 0
            if verbose:
                print(f"   Local annular background: NONE (no bck_{region_name} region)")

        # --- Three count-rate error contributions, combined in quadrature ---
        #
        # (1) From the user-drawn RMS region, scaled by sqrt(N_pix)
        count_rate_err_polygon = std_rms * np.sqrt(n_pixels)

        # (2) Local-background subtraction uncertainty (Bevington & Robinson):
        if n_ann > 0:
            count_rate_err_localbkg = std_rms * n_pixels / np.sqrt(n_ann)
        else:
            count_rate_err_localbkg = 0.0

        # (3) From the ERR extension (if available): per-pixel 1-sigma summed
        #     in quadrature over the aperture.
        if err_data is not None:
            err_region = err_data[mask]
            count_rate_err_errmap = float(np.sqrt(np.sum(err_region.astype(np.float64) ** 2)))
        else:
            count_rate_err_errmap = 0.0

        count_rate_error = float(np.sqrt(
            count_rate_err_polygon ** 2
            + count_rate_err_localbkg ** 2
            + count_rate_err_errmap ** 2
        ))

        if verbose:
            print(f"   Count rate error (electrons/s):")
            print(f"     from RMS-region polygon  : {count_rate_err_polygon:.6e}")
            print(f"     from local-bkg subtraction: {count_rate_err_localbkg:.6e}")
            print(f"     from ERR extension       : {count_rate_err_errmap:.6e}")
            print(f"     combined (quadrature)    : {count_rate_error:.6e}")
        
        # Calculate magnitude
        if count_rate > 0:
            magnitude = -2.5 * np.log10(count_rate) + zeropoint
            # Magnitude error propagation
            mag_error = 2.5 * count_rate_error / (count_rate * np.log(10))
            
            if verbose:
                print(f"   AB magnitude: {magnitude:.3f} ± {mag_error:.3f}")
            
            # Convert magnitude to microJansky using astropy units
            try:
                u_val_mag = magnitude * u.ABmag
                flux_ujy_obj = u_val_mag.to(1e-6 * u.Jy)
                flux_ujy = flux_ujy_obj.value

                # Convert magnitude error to flux error
                # Using: df/f = -ln(10)/2.5 * dmag,  equivalent here to df/f = dCR/CR.
                flux_ujy_error         = flux_ujy * (count_rate_error      / count_rate)
                flux_ujy_error_polygon = flux_ujy * (count_rate_err_polygon / count_rate)
                flux_ujy_error_errmap  = flux_ujy * (count_rate_err_errmap  / count_rate)

                if verbose:
                    print(f"   Flux: {flux_ujy:.2f} ± {flux_ujy_error:.2f} µJy")
                    print(f"     polygon contribution : {flux_ujy_error_polygon:.4f} µJy")
                    print(f"     errmap  contribution : {flux_ujy_error_errmap:.4f} µJy")
                
            except Exception as e:
                print(f"   ERROR converting magnitude to flux: {e}")
                flux_ujy = 0.0
                flux_ujy_error = 0.0
                flux_ujy_error_polygon = 0.0
                flux_ujy_error_errmap  = 0.0

        else:
            if verbose:
                print(f"   WARNING: Non-positive count rate ({count_rate:.6e})")
            magnitude = 99.999
            mag_error = 99.999
            flux_ujy = 0.0
            flux_ujy_error = 0.0
            flux_ujy_error_polygon = 0.0
            flux_ujy_error_errmap  = 0.0
        
        # Convert to milliJansky for consistency with JWST function
        flux_mjy               = flux_ujy * 1e-3
        flux_mjy_error         = flux_ujy_error * 1e-3
        flux_mjy_error_polygon = flux_ujy_error_polygon * 1e-3
        flux_mjy_error_errmap  = flux_ujy_error_errmap  * 1e-3
        
        if verbose:
            print(f"   Final flux: {flux_mjy:.6f} mJy")
            print(f"   =====================================")
        
        # Store results in same format as JWST function
        results.append({
            'region_name': region_name,
            'n_pixels': n_pixels,
            'flux_mJy': flux_mjy,
            'flux_mJy_error': flux_mjy_error,
            'flux_mJy_error_polygon': flux_mjy_error_polygon,
            'flux_mJy_error_errmap': flux_mjy_error_errmap,
            'flux_ujy': flux_ujy,
            'flux_ujy_error': flux_ujy_error,
            'flux_ujy_error_polygon': flux_ujy_error_polygon,
            'flux_ujy_error_errmap': flux_ujy_error_errmap,
            'magnitude': magnitude,
            'mag_error': mag_error
        })
        
        print(f"   FINAL: {region_name}: {count_rate:.2e} e/s → {flux_ujy:.2f} µJy ({flux_mjy:.6f} mJy), mag={magnitude:.3f}")
    
    return results


def get_nircam_cal_frac(filter_name):
    """
    Return the fractional 1-sigma absolute-flux calibration uncertainty
    for a given JWST/NIRCam filter, to be applied as `frac * |flux|` and
    added in quadrature to the per-aperture flux error.

    Reference (consult for the latest values):
        https://jwst-docs.stsci.edu/jwst-calibration-status/nircam-calibration-status/nircam-imaging-calibration-status

    Quoted from that page (NIRCam Imaging Calibration Status, photometric
    calibration section):

        "The absolute flux calibration uncertainties are <1% for most
         filters/detectors, and 1%-2% for the remainder. Four LW filters
         reach 2%-4%: F322W2 & F460M on module B and F430M & F460M on
         module A."

    The numbers in NIRCAM_FLUX_CAL_FRAC below use the conservative upper
    end of the documented range for each filter currently in use in this
    project. Module dependence is not encoded; if you want module-
    specific values, switch to keys like 'F322W2_B'. Update freely as the
    NIRCam calibration matures. Filters not in the table get 0.0
    (no contribution added; a notice is printed once).

    Parameters
    ----------
    filter_name : str or None
        Filter name from the FITS header (or user-provided prefix), e.g.
        'F150W2'. Case-insensitive.

    Returns
    -------
    frac : float
        Fractional 1-sigma absolute-flux calibration uncertainty.
        0.0 if `filter_name` is None or not in the lookup table.
    """
    if filter_name is None:
        return 0.0
    key = str(filter_name).strip().upper()
    return NIRCAM_FLUX_CAL_FRAC.get(key, 0.0)


# Lookup table consumed by get_nircam_cal_frac(). See its docstring for the
# source. Add additional filters here as the calibration evolves.
NIRCAM_FLUX_CAL_FRAC = {
    'F150W2': 0.02,   # SW wide-band; upper end of the 1-2% category
    'F322W2': 0.04,   # LW wide-band; module-B upper end of the 2-4% category
    # Examples to add if needed later (also in the 2-4% range per JWST docs):
    # 'F460M': 0.04,
    # 'F430M': 0.04,
}


def perform_jwst_photometry(data, masks, bunit, pixel_area_sr, std_rms,
                            err_data=None, cal_frac=0.0, filter_name=None,
                            local_bkg=None, n_annulus=None,
                            verbose=True):
    """
    Perform photometry on masked regions, with optional local-annulus
    background subtraction.

    Parameters:
    -----------
    data : numpy array
        Raw science image (NOT pre-subtracted in v10).
    masks : dict
        Dictionary of masks for each region (source name -> bool array).
    bunit : str
        Unit string from FITS header.
    pixel_area_sr : float
        Pixel area in steradians.
    std_rms : float
        Per-pixel RMS noise from RMS region (native units).
    err_data : numpy array or None
        Per-pixel 1-sigma error map (same shape/units as `data`).
        If provided, an additional aperture-level error contribution is
        computed as sqrt(sum(err_pix^2)) over the aperture, and combined
        in quadrature with the RMS-region-based error.
    cal_frac : float
        Fractional 1-sigma absolute-flux calibration uncertainty.
    filter_name : str or None
        Filter name, used only for printout/logging.
    local_bkg : dict {source_name: float} or None
        Per-source local background level (native units, per pixel). When
        a source has an entry, `local_bkg * n_aperture_pixels` is
        subtracted from the aperture sum, and the corresponding
        Bevington error term std_rms * n_pix / sqrt(n_annulus) is added
        in quadrature. Sources not in this dict get no subtraction.
    n_annulus : dict {source_name: int} or None
        Per-source annulus pixel count, used for the error propagation
        above. Required when local_bkg is supplied for that source.
    verbose : bool
        Print diagnostic information.

    Returns:
    --------
    results : list
        List of photometry results.
    """
    if local_bkg is None:
        local_bkg = {}
    if n_annulus is None:
        n_annulus = {}
    needs_pixel_area = 'MJY/SR' in bunit.upper() or '/SR' in bunit.upper()
    
    if needs_pixel_area:
        print(f"   Surface brightness unit detected - will use pixel area for conversion")
        test_val, conversion_factor = convert_to_microjansky(1.0, bunit, pixel_area_sr)
    else:
        print(f"   Per-pixel unit detected - no pixel area needed for conversion")
        test_val, conversion_factor = convert_to_microjansky(1.0, bunit)

    # One-time printout of the absolute-flux calibration uncertainty
    # (filter-dependent — see get_nircam_cal_frac and the JWST docs link
    # in its docstring).
    if cal_frac > 0.0:
        mag_floor_cal = 2.5 / np.log(10) * cal_frac
        print(f"   Absolute flux calibration uncertainty: {cal_frac*100:.2f}% "
              f"of flux  (filter: {filter_name or 'unspecified'})")
        print(f"     -> magnitude-error floor from calibration: ~{mag_floor_cal:.4f} mag")
    else:
        print(f"   No absolute flux calibration uncertainty applied "
              f"(filter: {filter_name or 'unspecified'} not in NIRCam cal table, "
              f"or explicitly disabled).")

    results = []
    
    for region_name, mask in masks.items():
        region_data = data[mask]
        n_pixels = len(region_data)
        
        if n_pixels == 0:
            print(f"   WARNING: {region_name} has no pixels!")
            continue
        
        if verbose:
            print(f"\n   === Diagnostics for {region_name} ===")
            print(f"   Number of pixels: {n_pixels}")
            print(f"   Raw data statistics (native units):")
            print(f"     Mean: {np.mean(region_data):.6e}")
            print(f"     Median: {np.median(region_data):.6e}")
            print(f"     Sum: {np.sum(region_data):.6e}")
            print(f"     Min: {np.min(region_data):.6e}")
            print(f"     Max: {np.max(region_data):.6e}")
        
        if needs_pixel_area:
            total_flux_native = np.sum(region_data)
            if verbose:
                print(f"   Surface brightness conversion:")
                print(f"     Total flux (native): {total_flux_native:.6e} {bunit}")
                print(f"     Pixel area: {pixel_area_sr:.6e} sr")
            total_flux_ujy, conv_factor = convert_to_microjansky(total_flux_native, bunit, pixel_area_sr)
            if verbose:
                print(f"     Conversion factor: {conv_factor:.6e}")
                print(f"     Total flux (µJy, before local bkg): {total_flux_ujy:.6e}")
        else:
            if verbose:
                print(f"   Per-pixel unit conversion:")
                print(f"     Unit: {bunit}")
                test_pixel = region_data[0] if len(region_data) > 0 else 1.0
                test_ujy, conv_factor = convert_to_microjansky(test_pixel, bunit)
                print(f"     Test pixel value: {test_pixel:.6e} {bunit}")
                print(f"     Test pixel in µJy: {test_ujy:.6e}")
                print(f"     Conversion factor: {conv_factor:.6e}")
            
            region_data_ujy, conv_factor = convert_to_microjansky(region_data, bunit)
            total_flux_ujy = np.sum(region_data_ujy)
            
            if verbose:
                print(f"     Sum before conversion: {np.sum(region_data):.6e} {bunit}")
                print(f"     Sum after conversion (before local bkg): {total_flux_ujy:.6e} µJy")

        # --- v10: subtract local annular background, if available ---
        if region_name in local_bkg:
            lb_per_pix = local_bkg[region_name]
            n_ann      = n_annulus.get(region_name, 0)
            # Native-unit total bkg in the aperture
            local_bkg_native = lb_per_pix * n_pixels
            # Convert to µJy with the same conv_factor that was applied to
            # the aperture sum above (works for both surface-brightness and
            # per-pixel unit branches).
            local_bkg_ujy = local_bkg_native * conv_factor
            total_flux_ujy_raw = total_flux_ujy
            total_flux_ujy = total_flux_ujy - local_bkg_ujy
            if verbose:
                print(f"   Local annular background:")
                print(f"     per-pixel level (native): {lb_per_pix:.6e} {bunit}")
                print(f"     annulus pixels         : {n_ann}")
                print(f"     subtracted total (µJy) : {local_bkg_ujy:.6e}")
                print(f"     flux after local sub   : {total_flux_ujy:.6e} µJy "
                      f"(was {total_flux_ujy_raw:.6e})")
        else:
            local_bkg_ujy = 0.0
            n_ann         = 0
            if verbose:
                print(f"   Local annular background: NONE (no bck_{region_name} region)")

        # --- Split the four error contributions and combine in quadrature ---
        #
        # (1) "Polygon" RMS-region contribution: assumes uniform per-pixel noise = std_rms
        flux_ujy_err_polygon = std_rms * np.sqrt(n_pixels) * conv_factor

        # (2) Local-background subtraction uncertainty (Bevington & Robinson form):
        #     the annulus median is uncertain at the level of std_rms / sqrt(n_annulus);
        #     multiplied by n_pixels because we subtracted that median once per pixel
        #     in the source aperture. Zero if no local bck was applied.
        if n_ann > 0:
            flux_ujy_err_localbkg = (std_rms * n_pixels / np.sqrt(n_ann)) * conv_factor
        else:
            flux_ujy_err_localbkg = 0.0

        # (3) ERR-map contribution: per-pixel 1-sigma uncertainty from the
        #     ERR extension of the i2d file, summed in quadrature over the aperture.
        #     Equivalent to sqrt(N) * RMS(err_pix), as the user described.
        if err_data is not None:
            err_region = err_data[mask]
            err_aperture_native = float(np.sqrt(np.sum(err_region.astype(np.float64) ** 2)))
            flux_ujy_err_errmap = err_aperture_native * conv_factor
            if verbose:
                err_rms_per_pix = float(np.sqrt(np.mean(err_region.astype(np.float64) ** 2))) if n_pixels > 0 else 0.0
                print(f"   ERR-map per-pixel RMS in aperture: {err_rms_per_pix:.6e} {bunit}")
                print(f"   ERR-map aperture error (quadrature sum, native): {err_aperture_native:.6e} {bunit}")
                print(f"   ERR-map aperture error (µJy): {flux_ujy_err_errmap:.6e}")
        else:
            flux_ujy_err_errmap = 0.0

        # (4) Absolute flux calibration uncertainty: cal_frac * |flux|.
        #     Filter-dependent constant, sourced from the JWST NIRCam
        #     calibration-status page (see get_nircam_cal_frac docstring).
        #     For faint apertures the background-subtracted total flux can
        #     be slightly negative; use |flux| so the term is well-defined.
        flux_ujy_err_calib = cal_frac * abs(total_flux_ujy)
        if verbose and cal_frac > 0.0:
            print(f"   Calibration error (µJy): {flux_ujy_err_calib:.6e}  "
                  f"({cal_frac*100:.2f}% of |flux|)")

        # Total uncertainty: sqrt(polygon^2 + localbkg^2 + errmap^2 + calib^2)
        flux_ujy_err = float(np.sqrt(
            flux_ujy_err_polygon ** 2
            + flux_ujy_err_localbkg ** 2
            + flux_ujy_err_errmap ** 2
            + flux_ujy_err_calib ** 2
        ))

        if verbose:
            print(f"   Flux error breakdown (µJy):")
            print(f"     from RMS-region polygon : {flux_ujy_err_polygon:.6e}")
            print(f"     from local-bkg subtraction: {flux_ujy_err_localbkg:.6e}")
            print(f"     from ERR extension      : {flux_ujy_err_errmap:.6e}")
            print(f"     from abs flux calib     : {flux_ujy_err_calib:.6e}")
            print(f"     combined (quadrature)   : {flux_ujy_err:.6e}")
            print(f"   =====================================")
        
        # v12: µJy -> mJy is 1e-3 (v11 used 1e-6, so the milliJy columns were in Jy)
        flux_mjy = total_flux_ujy * 1e-3
        flux_mjy_err = flux_ujy_err * 1e-3
        flux_mjy_err_polygon = flux_ujy_err_polygon * 1e-3
        flux_mjy_err_errmap = flux_ujy_err_errmap * 1e-3
        flux_mjy_err_calib = flux_ujy_err_calib * 1e-3

        if total_flux_ujy > 0:
            magnitude = -2.5 * np.log10(total_flux_ujy) + 23.9
            mag_error = 2.5 * flux_ujy_err / (total_flux_ujy * np.log(10))
        else:
            magnitude = 99.999
            mag_error = 99.999
        
        results.append({
            'region_name': region_name,
            'n_pixels': n_pixels,
            'flux_mJy': flux_mjy,
            'flux_mJy_error': flux_mjy_err,
            'flux_mJy_error_polygon': flux_mjy_err_polygon,
            'flux_mJy_error_errmap': flux_mjy_err_errmap,
            'flux_mJy_error_calib': flux_mjy_err_calib,
            'flux_ujy': total_flux_ujy,
            'flux_ujy_error': flux_ujy_err,
            'flux_ujy_error_polygon': flux_ujy_err_polygon,
            'flux_ujy_error_errmap': flux_ujy_err_errmap,
            'flux_ujy_error_calib': flux_ujy_err_calib,
            'magnitude': magnitude,
            'mag_error': mag_error
        })
        
        print(f"   FINAL: {region_name}: {total_flux_ujy:.2f} µJy ({flux_mjy:.4f} mJy), mag={magnitude:.3f}")
    
    return results

def compute_hst_zeropoint(PHOTFLAM,PHOTPLAM):
   #This should work for both WFC3 and ACS, it sure does for ACS
   #https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
   ZPAB=-2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408
   return ZPAB

def perform_decals_photometry(data, masks, bunit, pixel_area_sr, std_rms, header,
                              ivar_data=None, local_bkg=None, n_annulus=None,
                              verbose=True):
    """
    Perform aperture photometry on DECaLS Legacy Survey images.

    DECaLS single-band images are in nanomaggies (linear flux per pixel).
    The aperture flux is the **sum** of pixel values (not mean), because each
    pixel already carries its own flux contribution.

    Calibration keywords read from `header`:
      FLUXZP   : AB magnitude zero point. Formula: m = FLUXZP - 2.5*log10(flux).
                 Falls back to 22.5 (the standard DECaLS value) if absent.
      MWTRANS  : Milky Way dust transmission in [0, 1].  flux_corr = flux / MWTRANS.
                 If absent, a warning is printed and no correction is applied.
      FILTER   : Band name (g, r, i, z), used for logging only.
      NMAGYNJ  : nanomaggy-to-nanoJansky factor. Falls back to 3.631 if absent.

    Error budget (all terms in nanomaggies, combined in quadrature):
      (1) RMS-region polygon term : std_rms * sqrt(N_pix)
      (2) Bevington local-bkg term: std_rms * N_pix / sqrt(N_annulus)  [if bck_]
      (3) ivar-map term           : sqrt( sum(1 / ivar_pix) ) over valid pixels
                                    where valid means ivar_pix > 0.
                                    Zero if no ivar map is supplied.

    Note: the error on the MW-corrected flux is scaled by 1/MWTRANS together
    with the flux itself (linear propagation through division by a constant).

    Parameters
    ----------
    data : numpy array
        Science image in nanomaggies (NOT pre-subtracted in v11).
    masks : dict {str: bool array}
        Aperture masks for each photometry source.
    bunit : str
        BUNIT string from header, used for logging.
    pixel_area_sr : float
        Pixel area in steradians (not used for nanomaggy flux; kept for
        API consistency).
    std_rms : float
        Per-pixel RMS noise from the user-drawn RMS region (nanomaggies).
    header : FITS header
        Science image header (supplies FLUXZP, MWTRANS, NMAGYNJ, FILTER).
    ivar_data : numpy array or None
        Inverse-variance map (nanomaggies^-2 per pixel).  When provided,
        the ivar-map error term replaces the need for an ERR extension.
    local_bkg : dict {str: float} or None
        Per-source local background level (nanomaggies per pixel).
    n_annulus : dict {str: int} or None
        Per-source annulus pixel count (for Bevington error term).
    verbose : bool
        Print per-source diagnostic breakdown.

    Returns
    -------
    results : list of dict
        One dict per source with keys:
          region_name, n_pixels,
          flux_nanomaggy, flux_nanomaggy_error,
          flux_nanomaggy_error_polygon, flux_nanomaggy_error_localbkg,
          flux_nanomaggy_error_ivar,
          flux_nJy, flux_nJy_error,
          flux_ujy, flux_ujy_error,
          flux_mJy, flux_mJy_error,
          flux_mJy_error_polygon, flux_mJy_error_errmap,
          magnitude, mag_error,
          mwtrans_applied (float; 1.0 if no MWTRANS correction).
    """
    if local_bkg is None:
        local_bkg = {}
    if n_annulus is None:
        n_annulus = {}

    # ------------------------------------------------------------------
    # Read calibration keywords from header
    # ------------------------------------------------------------------
    fluxzp = header.get('FLUXZP', None)
    if fluxzp is None:
        fluxzp = 22.5
        print(f"   WARNING: FLUXZP not found in header; using default value 22.5")
    else:
        print(f"   FLUXZP = {fluxzp:.4f} (AB mag zeropoint)")

    mwtrans = header.get('MWTRANS', None)
    if mwtrans is None:
        print(f"   WARNING: MWTRANS not found in header; MW extinction correction "
              f"will NOT be applied.")
        mwtrans = 1.0         # no correction; flag below stays False
        mwtrans_applied = 1.0
        do_mwtrans = False
    else:
        if not (0.0 < mwtrans <= 1.0):
            print(f"   WARNING: MWTRANS = {mwtrans:.6f} is outside (0, 1]; "
                  f"applying anyway but check the header.")
        print(f"   MWTRANS = {mwtrans:.6f} (MW dust transmission; "
              f"flux will be divided by this value)")
        mwtrans_applied = float(mwtrans)
        do_mwtrans = True

    # nanomaggy → nanoJansky conversion factor (header or physical constant)
    nmag2njy = float(header.get('NMAGYNJ', 3.631))
    if verbose:
        print(f"   NMAGYNJ = {nmag2njy:.6f} nJy per nanomaggy")

    filter_name = header.get('FILTER', 'unknown')
    print(f"   Filter: {filter_name}")

    results = []

    for region_name, mask in masks.items():
        region_data = data[mask]
        n_pixels = int(len(region_data))

        if n_pixels == 0:
            print(f"   WARNING: {region_name} has no pixels!")
            continue

        if verbose:
            print(f"\n   === DECaLS Photometry for {region_name} ===")
            print(f"   Number of pixels: {n_pixels}")
            print(f"   Raw data statistics (nanomaggies):")
            print(f"     Mean  : {np.mean(region_data):.6e}")
            print(f"     Median: {np.median(region_data):.6e}")
            print(f"     Sum   : {np.sum(region_data):.6e}")

        # --- Raw aperture sum (nanomaggies) ---
        flux_raw = float(np.sum(region_data))
        if verbose:
            print(f"   Aperture sum (raw): {flux_raw:.6e} nanomaggy")

        # --- Local annular background subtraction ---
        if region_name in local_bkg:
            lb_per_pix = local_bkg[region_name]
            n_ann      = n_annulus.get(region_name, 0)
            flux_bkg   = lb_per_pix * n_pixels
            flux       = flux_raw - flux_bkg
            if verbose:
                print(f"   Local annular background:")
                print(f"     per-pixel level : {lb_per_pix:.6e} nanomaggy")
                print(f"     annulus pixels  : {n_ann}")
                print(f"     subtracted total: {flux_bkg:.6e} nanomaggy")
                print(f"   Flux after local subtraction: {flux:.6e} nanomaggy")
        else:
            flux  = flux_raw
            n_ann = 0
            if verbose:
                print(f"   Local annular background: NONE (no bck_{region_name} region)")

        # --- MW extinction correction ---
        if do_mwtrans:
            flux_ext_corr = flux / mwtrans_applied
            if verbose:
                print(f"   MW extinction correction (÷ {mwtrans_applied:.6f}): "
                      f"{flux:.6e} → {flux_ext_corr:.6e} nanomaggy")
        else:
            flux_ext_corr = flux

        # --- Error budget (nanomaggies, pre-MW correction) ---
        # (1) RMS-region polygon
        err_polygon = std_rms * np.sqrt(n_pixels)

        # (2) Bevington local-background subtraction uncertainty
        if n_ann > 0:
            err_localbkg = std_rms * n_pixels / np.sqrt(n_ann)
        else:
            err_localbkg = 0.0

        # (3) ivar-map contribution: sqrt( sum(1/ivar) ) over valid pixels
        if ivar_data is not None:
            ivar_region = ivar_data[mask].astype(np.float64)
            valid = ivar_region > 0
            n_invalid = int(np.sum(~valid))
            if n_invalid > 0:
                print(f"   WARNING: {region_name}: {n_invalid} aperture pixel(s) "
                      f"have ivar=0 and are excluded from the ivar error term.")
            if np.any(valid):
                err_ivar = float(np.sqrt(np.sum(1.0 / ivar_region[valid])))
            else:
                print(f"   WARNING: {region_name}: all ivar pixels are zero; "
                      f"ivar error term set to 0.")
                err_ivar = 0.0
            if verbose:
                ivar_rms = float(np.sqrt(np.mean(1.0 / ivar_region[valid]))) if np.any(valid) else 0.0
                print(f"   ivar-map: per-pixel noise RMS in aperture = {ivar_rms:.6e} nanomaggy")
                print(f"   ivar-map aperture error (quadrature sum)   = {err_ivar:.6e} nanomaggy")
        else:
            err_ivar = 0.0

        # Combined (quadrature), still in pre-MW-correction units.
        # After MW correction the flux is scaled by 1/MWTRANS, so all error
        # terms scale by the same factor.
        err_total_raw = float(np.sqrt(err_polygon**2 + err_localbkg**2 + err_ivar**2))
        err_total     = err_total_raw / mwtrans_applied   # propagate through ÷ MWTRANS

        if verbose:
            print(f"   Error breakdown (nanomaggies, before MW correction):")
            print(f"     from RMS-region polygon    : {err_polygon:.6e}")
            print(f"     from local-bkg subtraction : {err_localbkg:.6e}")
            print(f"     from ivar map              : {err_ivar:.6e}")
            print(f"     combined (pre-correction)  : {err_total_raw:.6e}")
            if do_mwtrans:
                print(f"     combined (post-correction) : {err_total:.6e}")

        # --- AB magnitude ---
        if flux_ext_corr > 0:
            magnitude = fluxzp - 2.5 * np.log10(flux_ext_corr)
            mag_error = 2.5 * err_total / (flux_ext_corr * np.log(10))
        else:
            if verbose:
                print(f"   WARNING: Non-positive MW-corrected flux ({flux_ext_corr:.6e}); "
                      f"setting magnitude to 99.999")
            magnitude = 99.999
            mag_error = 99.999

        # --- Unit conversions ---
        flux_njy       = flux_ext_corr * nmag2njy          # nanoJansky
        flux_njy_error = err_total     * nmag2njy

        flux_ujy       = flux_njy       * 1e-3              # microJansky
        flux_ujy_error = flux_njy_error * 1e-3

        flux_mjy       = flux_ujy       * 1e-3              # milliJansky (v12: was 1e-6)
        flux_mjy_error = flux_ujy_error * 1e-3

        # Decompose error contributions in mJy for table consistency (v12: nJy -> mJy is 1e-6, was 1e-9)
        flux_mjy_error_polygon  = (err_polygon  / mwtrans_applied) * nmag2njy * 1e-6
        flux_mjy_error_errmap   = (err_ivar     / mwtrans_applied) * nmag2njy * 1e-6

        if verbose:
            print(f"   Results:")
            print(f"     Flux (nanomaggy)  : {flux_ext_corr:.6e} ± {err_total:.6e}")
            print(f"     Flux (nJy)        : {flux_njy:.4f} ± {flux_njy_error:.4f}")
            print(f"     Flux (µJy)        : {flux_ujy:.6e} ± {flux_ujy_error:.6e}")
            print(f"     AB magnitude      : {magnitude:.3f} ± {mag_error:.3f}")
            print(f"   =====================================")

        print(f"   FINAL: {region_name}: {flux_ext_corr:.4e} nanomaggy "
              f"= {flux_njy:.2f} nJy, mag={magnitude:.3f}±{mag_error:.3f}"
              + (f"  [MW corr: ÷{mwtrans_applied:.4f}]" if do_mwtrans else ""))

        results.append({
            'region_name'                 : region_name,
            'n_pixels'                    : n_pixels,
            # Native DECaLS units
            'flux_nanomaggy'              : float(flux_ext_corr),
            'flux_nanomaggy_error'        : float(err_total),
            'flux_nanomaggy_error_polygon': float(err_polygon / mwtrans_applied),
            'flux_nanomaggy_error_localbkg': float(err_localbkg / mwtrans_applied),
            'flux_nanomaggy_error_ivar'   : float(err_ivar / mwtrans_applied),
            # nanoJansky (JWST-native unit, convenient for comparison)
            'flux_nJy'                    : float(flux_njy),
            'flux_nJy_error'              : float(flux_njy_error),
            # Wider unit ladder (for table consistency with JWST/HST rows)
            'flux_ujy'                    : float(flux_ujy),
            'flux_ujy_error'              : float(flux_ujy_error),
            'flux_ujy_error_polygon'      : float(flux_mjy_error_polygon * 1e3),
            'flux_ujy_error_errmap'       : float(flux_mjy_error_errmap  * 1e3),
            'flux_mJy'                    : float(flux_mjy),
            'flux_mJy_error'              : float(flux_mjy_error),
            'flux_mJy_error_polygon'      : float(flux_mjy_error_polygon),
            'flux_mJy_error_errmap'       : float(flux_mjy_error_errmap),
            'magnitude'                   : float(magnitude),
            'mag_error'                   : float(mag_error),
            'mwtrans_applied'             : float(mwtrans_applied),
        })

    return results


def perform_euclid_photometry(data, masks, header, std_rms, zeropoint=None,
                              err_data=None, cal_frac=0.0, filter_name=None,
                              local_bkg=None, n_annulus=None, verbose=True):
    """
    v12: photometry on Euclid MER mosaics / cutouts.

    MER pixel values are linear count rates (VIS in ADU/s, NISP Y/J/H in e-/s) with an AB zeropoint
    MAGZERO in the header:  m_AB = MAGZERO - 2.5 log10(sum of pixels).  Each pixel is therefore
    converted to µJy with the constant factor 10**(-0.4 * (MAGZERO - 23.9)), and the measurement is
    handed to perform_jwst_photometry() in µJy.  Every term in that function (aperture sum, local
    background, RMS-region, Bevington, ERR-map and calibration errors) is linear in the pixel values,
    so this gives exactly the same result as doing the arithmetic in native units.

    Parameters are as for perform_jwst_photometry, plus
    zeropoint : float or None
        AB zeropoint; overrides the header MAGZERO when given.
    """
    if zeropoint is None:
        if 'MAGZERO' not in header:
            print("   ERROR: no MAGZERO in the header and no --zeropoint given; cannot calibrate Euclid data.")
            sys.exit(1)
        zeropoint = float(header['MAGZERO'])
        print(f"   Euclid MAGZERO (header) = {zeropoint:.4f}")
    else:
        print(f"   Euclid zeropoint (override) = {zeropoint:.4f}")
    conv = 10 ** (-0.4 * (zeropoint - 23.9))          # µJy per (ADU/s or e-/s)
    print(f"   Native units: {header.get('BUNIT', 'count rate')};  1 unit = {conv:.6e} µJy")
    scaled_bkg = {k: v * conv for k, v in (local_bkg or {}).items()}
    return perform_jwst_photometry(data * conv, masks, 'MICROJANSKY', None, std_rms * conv,
                                   err_data=None if err_data is None else err_data * conv,
                                   cal_frac=cal_frac, filter_name=filter_name,
                                   local_bkg=scaled_bkg, n_annulus=n_annulus, verbose=verbose)


def get_hst_FLAM_PLAM(meas_sci_im):
   hdul = fits.open(meas_sci_im)
   hdr = hdul[0].header
   PHOTFLAM=hdr['PHOTFLAM']
   PHOTPLAM=hdr['PHOTPLAM']
   return PHOTFLAM,PHOTPLAM


def count_rate_to_ab_mag(count_rate, zeropoint):
    """
    Convert count rate to AB magnitude.
    
    Parameters:
    -----------
    count_rate : float or array-like
        Count rate in electrons/s
    zeropoint : float
        Photometric zeropoint for the instrument/filter combination
        
    Returns:
    --------
    magnitude : float or array-like
        AB magnitude
        
    Notes:
    ------
    Uses the standard formula: mag = -2.5 * log10(count_rate) + zeropoint
    Returns NaN for non-positive count rates.
    """
    count_rate = np.asarray(count_rate)
    
    # Handle non-positive count rates
    if np.any(count_rate <= 0):
        magnitude = np.full_like(count_rate, np.nan, dtype=float)
        valid_mask = count_rate > 0
        magnitude[valid_mask] = -2.5 * np.log10(count_rate[valid_mask]) + zeropoint
    else:
        magnitude = -2.5 * np.log10(count_rate) + zeropoint
    
    return magnitude



def compute_mask_centroids_wcs(masks, wcs):
    """
    Compute the unweighted pixel centroid of each boolean aperture mask,
    then convert to sky coordinates (RA, Dec in degrees) using the WCS.

    The "centroid" here is the arithmetic mean of (x, y) pixel coordinates
    of all True pixels in the mask. This is the natural geometric center
    of the aperture and is exactly the point that minimizes the sum of
    squared distances to all aperture pixels (the "barycenter" the user
    asked for). It is independent of the underlying image flux, so it
    tracks the aperture you drew rather than the brightest blob inside it.

    Notes
    -----
    For very large apertures (many arcsec across) the pixel-space mean and
    the sky-space mean can differ slightly due to the tangent-plane
    projection. For typical galaxy-cluster object apertures (sub-arcsec to
    a few arcsec) the difference is negligible.

    Parameters
    ----------
    masks : dict
        Mapping {region_name: 2D boolean array}.
    wcs : astropy.wcs.WCS
        WCS used to convert pixel -> sky.

    Returns
    -------
    dict
        Mapping {region_name: (ra_deg, dec_deg)}. NaN pair if the mask is empty.
    """
    out = {}
    for name, mask in masks.items():
        ys, xs = np.where(mask)
        if len(xs) == 0:
            out[name] = (float('nan'), float('nan'))
            continue
        # NB: astropy WCS.pixel_to_world takes (x, y) order; np.where returns (y, x).
        x_c = float(np.mean(xs))
        y_c = float(np.mean(ys))
        sky = wcs.pixel_to_world(x_c, y_c)
        out[name] = (float(sky.ra.deg), float(sky.dec.deg))
    return out


def save_results(results, header, data, photometry_masks, median_global,
                 std_global, std_rms, prefix='',
                 local_bkg=None, n_annulus=None, annulus_masks=None,
                 bunit=''):
    """
    Save photometry results and create visualizations.

    Parameters:
    -----------
    results : list
        Photometry results.
    header : FITS header
        Image header.
    data : numpy array
        Raw image data (no global background subtraction in v10).
    photometry_masks : dict
        Dictionary of source masks (name -> bool array).
    median_global : float
        Representative background level used for display stretch only.
    std_global : float
        Representative std used for display stretch only.
    std_rms : float
        Per-pixel RMS used for display stretch in the locally-subtracted panel.
    prefix : str
        Prefix to prepend to all output filenames.
    local_bkg : dict or None
        Per-source local background per pixel (native units). Sources not in
        this dict get no local subtraction in the middle plot panel.
    n_annulus : dict or None
        Per-source annulus pixel count (for annotation).
    annulus_masks : dict or None
        Per-source boolean annulus masks. If supplied, the middle panel will
        overlay the annulus contour.
    bunit : str
        Unit string from the FITS header, used in the bkg-level annotation.
    """
    if local_bkg is None:
        local_bkg = {}
    if n_annulus is None:
        n_annulus = {}
    if annulus_masks is None:
        annulus_masks = {}
    print("\n10. Saving results...")
    file_prefix = f"{prefix}_" if prefix else ''
    
    if len(results) > 0:
        # Create astropy table.
        # NOTE: HST results don't have the *_calib fields; use .get() with 0.0 default.
        # DECaLS results carry flux_nanomaggy and flux_nJy; JWST/HST rows get NaN.
        result_table = Table([
            [prefix] * len(results),
            [r['region_name'] for r in results],
            [r.get('ra_deg',  float('nan')) for r in results],
            [r.get('dec_deg', float('nan')) for r in results],
            [r['n_pixels'] for r in results],
            # DECaLS native units
            [r.get('flux_nanomaggy',       float('nan')) for r in results],
            [r.get('flux_nanomaggy_error', float('nan')) for r in results],
            [r.get('flux_nJy',             float('nan')) for r in results],
            [r.get('flux_nJy_error',       float('nan')) for r in results],
            [r.get('mwtrans_applied',      float('nan')) for r in results],
            # mJy/µJy ladder (populated for all telescopes)
            [r['flux_mJy'] for r in results],
            [r['flux_mJy_error'] for r in results],
            [r['flux_mJy_error_polygon'] for r in results],
            [r['flux_mJy_error_errmap'] for r in results],
            [r.get('flux_mJy_error_calib', 0.0) for r in results],
            [r['flux_ujy'] for r in results],
            [r['flux_ujy_error'] for r in results],
            [r['flux_ujy_error_polygon'] for r in results],
            [r['flux_ujy_error_errmap'] for r in results],
            [r.get('flux_ujy_error_calib', 0.0) for r in results],
            [r['magnitude'] for r in results],
            [r['mag_error'] for r in results]
        ], names=['prefix', 'region_name', 'ra_deg', 'dec_deg', 'n_pixels',
                  'flux_nanomaggy', 'flux_nanomaggy_error',
                  'flux_nJy', 'flux_nJy_error',
                  'mwtrans_applied',
                  'flux_milliJy', 'flux_milliJy_error',
                  'flux_milliJy_error_polygon', 'flux_milliJy_error_errmap',
                  'flux_milliJy_error_calib',
                  'flux_microJy', 'flux_microJy_error',
                  'flux_microJy_error_polygon', 'flux_microJy_error_errmap',
                  'flux_microJy_error_calib',
                  'magnitude', 'mag_error'])
        
        # Save as FITS
        result_table.write(f'{file_prefix}custom_photometry_results.fits', overwrite=True)
        print(f"   ✓ Results saved as: {file_prefix}custom_photometry_results.fits")
        
        # Save as ASCII
        result_table.write(f'{file_prefix}custom_photometry_results.txt', format='ascii.fixed_width_two_line', overwrite=True)
        print(f"   ✓ Results saved as: {file_prefix}custom_photometry_results.txt")
        
        # Print summary
        print(f"\n   PHOTOMETRY SUMMARY:")
        print(f"   ==================")
        print(result_table)
    else:
        print("   ERROR: No photometry results to save!")
        return
    
    print("\n11. Saving masks and creating detailed visualization...")
    
    # Save individual masks
    mask_header = header.copy()
    mask_header['BUNIT'] = 'dimensionless'
    mask_header.add_comment('Mask: 1=source pixels, 0=background pixels')
    
    for region_name, mask in photometry_masks.items():
        mask_filename = f'{file_prefix}mask_{region_name}.fits'
        fits.writeto(mask_filename, mask.astype(np.uint8), mask_header, overwrite=True)
        print(f"   ✓ Mask for {region_name} saved as: {mask_filename}")
    
    # Create visualizations
    for region_name, mask in photometry_masks.items():
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        padding = 30
        x_min = max(0, np.min(x_indices) - padding)
        x_max = min(data.shape[1], np.max(x_indices) + padding)
        y_min = max(0, np.min(y_indices) - padding)
        y_max = min(data.shape[0], np.max(y_indices) + padding)

        zoom_original = data[y_min:y_max, x_min:x_max]
        zoom_mask     = mask[y_min:y_max, x_min:x_max]

        # Middle-panel data: locally-subtracted cutout. If this source has a
        # bck_ region, we subtract a uniform constant (= local_bkg per pixel)
        # over the entire cutout, so the source itself sits on a ~zero sky.
        has_local_bkg = region_name in local_bkg
        if has_local_bkg:
            lb_val       = local_bkg[region_name]
            n_ann        = n_annulus.get(region_name, 0)
            zoom_bkg_sub = zoom_original - lb_val
        else:
            lb_val       = 0.0
            n_ann        = 0
            zoom_bkg_sub = zoom_original  # nothing to subtract; show raw

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # --- Panel 1: original cutout ---
        im1 = axes[0].imshow(zoom_original, origin='lower', cmap='gray',
                             vmin=median_global - 2 * std_global,
                             vmax=median_global + 10 * std_global)
        axes[0].set_title(f'{region_name} - Original Image')
        axes[0].set_xlabel('X pixel (relative)')
        axes[0].set_ylabel('Y pixel (relative)')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # --- Panel 2: locally-subtracted cutout, annulus overlay, bkg label ---
        im2 = axes[1].imshow(zoom_bkg_sub, origin='lower', cmap='gray',
                             vmin=-3 * std_rms, vmax=10 * std_rms)
        if has_local_bkg:
            axes[1].set_title(f'{region_name} - Local Bkg Subtracted')
        else:
            axes[1].set_title(f'{region_name} - (no bck_ region; raw shown)')
        axes[1].set_xlabel('X pixel (relative)')
        axes[1].set_ylabel('Y pixel (relative)')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # Overlay the annulus contour (in cutout coords)
        if region_name in annulus_masks:
            zoom_annulus = annulus_masks[region_name][y_min:y_max, x_min:x_max]
            if zoom_annulus.any():
                axes[1].contour(zoom_annulus.astype(float), levels=[0.5],
                                colors='lime', linewidths=1.2)

        # Annotate the bkg level on the middle panel. White text on a
        # semi-transparent dark box so it's readable against any image.
        unit_str = bunit if bunit else 'native'
        if has_local_bkg:
            label = (f"local bkg = {lb_val:.4e} {unit_str}/pix\n"
                     f"annulus N = {n_ann} pix")
        else:
            label = "no local bkg applied"
        axes[1].text(
            0.03, 0.97, label,
            transform=axes[1].transAxes,
            ha='left', va='top',
            fontsize=9, color='white', family='monospace',
            bbox=dict(facecolor='black', alpha=0.55, edgecolor='none',
                      boxstyle='round,pad=0.3'),
        )

        # --- Panel 3: source mask ---
        im3 = axes[2].imshow(zoom_mask, origin='lower', cmap='RdYlBu_r',
                             vmin=0, vmax=1)
        axes[2].set_title(f'{region_name} - Mask ({np.sum(zoom_mask)} pixels)')
        axes[2].set_xlabel('X pixel (relative)')
        axes[2].set_ylabel('Y pixel (relative)')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

        plt.tight_layout()

        plot_filename = f'{file_prefix}photometry_{region_name}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"   ✓ Individual plot for {region_name} saved as: {plot_filename}")
    
    # Create overview image
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(data, origin='lower', cmap='gray',
                   vmin=median_global-2*std_global, vmax=median_global+10*std_global)
    ax.set_title('Overview: All Regions')
    
    for region_name, mask in photometry_masks.items():
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            center_x = np.mean(x_indices)
            center_y = np.mean(y_indices)
            ax.text(center_x, center_y, region_name, 
                    color='red', fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}custom_photometry_overview.png', dpi=150, bbox_inches='tight')
    
    print(f"   ✓ Overview image saved as: {file_prefix}custom_photometry_overview.png")


def compute_local_backgrounds(data, bck_regions_pix, photometry_masks,
                              sigma=3.0, maxiters=5, verbose=True):
    """
    Compute per-source local background levels from bck_<name> annuli.

    For each entry in bck_regions_pix, the annulus mask is the bck mask with
    the corresponding source mask (from photometry_masks) removed. This
    yields a true annulus even if the bck polygon overlaps the source
    polygon, which is the common case (the bck_ polygon is drawn around the
    source as a slightly enlarged copy of the same shape).

    Parameters
    ----------
    data : 2D ndarray
        Science image (native units).
    bck_regions_pix : list of dict
        Pixel-coordinate bck regions. Each must have:
          - 'name'  : original DS9 label, e.g. 'bck_laal1'
          - 'target': source name this bck applies to, e.g. 'laal1'
          - plus the geometry needed by create_region_mask.
    photometry_masks : dict {source_name: 2D bool ndarray}
        Source masks for the photometry regions.
    sigma : float
        Sigma-clipping threshold used when computing the annulus median.
    maxiters : int
        Maximum sigma-clipping iterations.
    verbose : bool
        Print per-source diagnostics.

    Returns
    -------
    local_bkg : dict {source_name: float}
        Per-pixel local background level (native data units) per source.
        Sources with no matching bck region are NOT in this dict.
    n_annulus : dict {source_name: int}
        Number of annulus pixels used (after source subtraction and sigma
        clipping by inclusion, not by rejection — i.e. the raw annulus size).
    annulus_masks : dict {source_name: 2D bool ndarray}
        Per-source annulus mask used.
    annulus_label_image : 2D int ndarray
        Label image with integer i for annulus of i-th photometry source,
        0 elsewhere. Useful for QA in DS9.
    """
    local_bkg = {}
    n_annulus = {}
    annulus_masks = {}
    annulus_label_image = np.zeros(data.shape, dtype=np.int32)

    # Establish a stable integer label per source name based on the order of
    # photometry_masks (a dict preserves insertion order in Py3.7+).
    label_lookup = {name: i + 1 for i, name in enumerate(photometry_masks.keys())}

    for bck in bck_regions_pix:
        target = bck['target']
        if target not in photometry_masks:
            # The categorize step already warned about orphan bck regions,
            # but skip defensively here too.
            print(f"   SKIP: bck_{target} has no matching source mask.")
            continue

        bck_mask    = create_region_mask(bck, data.shape)
        source_mask = photometry_masks[target]
        annulus    = bck_mask & ~source_mask
        n_pix      = int(annulus.sum())

        if n_pix == 0:
            print(f"   WARNING: annulus for '{target}' is empty after source "
                  f"removal; skipping local subtraction for this source.")
            continue

        annulus_pixels = data[annulus]
        mean_b, median_b, std_b = sigma_clipped_stats(
            annulus_pixels, sigma=sigma, maxiters=maxiters)

        local_bkg[target]     = float(median_b)
        n_annulus[target]     = n_pix
        annulus_masks[target] = annulus
        annulus_label_image[annulus] = label_lookup[target]

        if verbose:
            print(f"   {target:<12s}: bck pixels={int(bck_mask.sum()):>6d}, "
                  f"src pixels={int(source_mask.sum()):>5d}, "
                  f"annulus={n_pix:>6d}  |  "
                  f"median={median_b:.6e}  std={std_b:.6e}")

    return local_bkg, n_annulus, annulus_masks, annulus_label_image


def custom_aperture_photometry(image_name, region_file, science_ext=0, verbose=True,
                               telescope_override=None, prefix='', err_ext='ERR',
                               cal_frac_override=None, ivar_name=None,
                               zeropoint_override=None):
    """
    Master function to perform custom aperture photometry.
    
    Parameters:
    -----------
    image_name : str
        Path to FITS file
    region_file : str
        Path to DS9 region file
    science_ext : int
        FITS extension for science data (default: 0)
    verbose : bool
        Print diagnostic information (default: True)
    telescope_override : str or None
        If provided ('JWST' or 'HST'), overrides the TELESCOP keyword in the header
    prefix : str
        If provided, prepended to all output filenames (e.g. 'F390W')
    err_ext : int, str, or None
        FITS extension (by index or name) containing the per-pixel 1-sigma
        ERR map. Defaults to 'ERR' (works for JWST i2d products). Pass None
        to disable the ERR-map error contribution entirely.  For DECaLS
        images, pass None here and supply `ivar_name` instead.
    cal_frac_override : float or None
        If not None, force the absolute-flux calibration uncertainty to this
        fractional value (e.g. 0.02 for 2%) regardless of filter. Otherwise
        the value is looked up via get_nircam_cal_frac() from the FITS
        FILTER keyword (or the prefix, as a fallback).
    ivar_name : str or None
        Path to a DECaLS inverse-variance map FITS file (nanomaggies^-2 per
        pixel).  Used only when telescop == 'DECALS'/'DECAM'.  The science
        extension (index 0) of this file is loaded.  Pass None to skip the
        ivar error contribution (only the RMS-region polygon term will apply).

    Returns:
    --------
    results : list
        Photometry results
    """
    print("=== Custom Aperture Photometry ===")
    print("Reading DS9 regions and performing photometry...\n")
    
    # Load FITS image (and ERR extension if available)
    data, header, wcs, err_data = load_fits_image(image_name, science_ext, err_ext=err_ext)

    # --- Load DECaLS inverse-variance map (if provided) ---
    ivar_data = None
    if ivar_name is not None:
        try:
            with fits.open(ivar_name) as hdul_ivar:
                ivar_data = hdul_ivar[0].data.copy().astype(np.float64)
            if ivar_data.shape != data.shape:
                print(f"   WARNING: ivar map shape {ivar_data.shape} does not match "
                      f"science image shape {data.shape}; ignoring ivar map.")
                ivar_data = None
            else:
                # Replace NaN/inf with 0 (no information → excluded from error sum)
                bad = ~np.isfinite(ivar_data) | (ivar_data < 0)
                if bad.any():
                    print(f"   WARNING: {int(bad.sum())} ivar pixel(s) are NaN/inf/<0; "
                          f"replaced with 0.")
                    ivar_data[bad] = 0.0
                print(f"   Loaded ivar map from: {ivar_name}  (shape {ivar_data.shape})")
        except Exception as e:
            print(f"   WARNING: Could not load ivar map '{ivar_name}': {e}")
            print(f"   Proceeding without ivar error contribution.")
            ivar_data = None

    # --- Resolve filter name for calibration-uncertainty lookup ---
    # JWST i2d files keep FILTER in the PRIMARY header; if not present in
    # the science header, peek there. If still nothing, fall back to the
    # user-provided prefix (which is typically the filter name).
    filter_name = header.get('FILTER', None)
    if not filter_name:
        try:
            with fits.open(image_name) as _hdul:
                filter_name = _hdul[0].header.get('FILTER', None)
        except Exception:
            filter_name = None
    if not filter_name and prefix:
        filter_name = prefix
    if filter_name:
        filter_name = str(filter_name).strip()
    
    # Calculate pixel scale
    pixel_scale_arcsec, pixel_area_sr = calculate_pixel_scale(header, wcs)
    
    # Parse DS9 regions
    regions = parse_ds9_regions(region_file)
    
    # Categorize regions into RMS / bck / photometry
    rms_regions, bck_regions, photometry_regions = categorize_regions(regions)

    # Convert to pixel coordinates. The `target` field of bck regions is
    # re-attached after conversion (sky_to_pixel_region returns a fresh dict
    # with only the geometric fields and 'name').
    print("\n4. Converting sky coordinates to pixels...")
    rms_regions_pix = [sky_to_pixel_region(r, wcs, pixel_scale_arcsec)
                       for r in rms_regions]
    bck_regions_pix = []
    BCK_NAME_RE = re.compile(r'^bck_(.+)$', re.IGNORECASE)
    for r in bck_regions:
        r_pix = sky_to_pixel_region(r, wcs, pixel_scale_arcsec)
        m = BCK_NAME_RE.match(r_pix['name'])
        r_pix['target'] = m.group(1) if m else r.get('target')
        bck_regions_pix.append(r_pix)
    photometry_regions_pix = [sky_to_pixel_region(region, wcs, pixel_scale_arcsec)
                              for region in photometry_regions]
    print(f"   Converted {len(photometry_regions_pix)} photometry, "
          f"{len(bck_regions_pix)} bck, and "
          f"{len(rms_regions_pix)} RMS region(s) to pixels")

    # Create masks for sources and RMS regions
    print("\n5. Creating masks for each region...")
    rms_masks = [create_region_mask(r, data.shape) for r in rms_regions_pix]
    photometry_masks = {}
    for region in photometry_regions_pix:
        mask = create_region_mask(region, data.shape)
        photometry_masks[region['name']] = mask
        print(f"   {region['name']}: {int(np.sum(mask))} pixels")
    for r, m in zip(rms_regions_pix, rms_masks):
        print(f"   {r['name']} (RMS): {int(np.sum(m))} pixels")

    # -------------------------------------------------------------------------
    # 6. Compute per-pixel noise (std_rms) from the user's DS9 RMS regions.
    #    Same logic as v7-v9: sigma-clipped std in each rms_<n> region, then
    #    take the median across regions for robustness.
    # -------------------------------------------------------------------------
    print("\n6. Estimating per-pixel noise from RMS region(s)...")
    per_region_stds = []
    for r, m in zip(rms_regions_pix, rms_masks):
        rms_data = data[m]
        if len(rms_data) == 0:
            print(f"   WARNING: RMS region {r['name']} contains no pixels - skipping!")
            continue
        _, _, std_i = sigma_clipped_stats(rms_data, sigma=3.0, maxiters=5)
        per_region_stds.append(std_i)
        print(f"   {r['name']}: sigma = {std_i:.6e}  ({len(rms_data)} pixels)")

    if not per_region_stds:
        print("   ERROR: All RMS regions are empty!")
        sys.exit(1)

    std_rms = float(np.median(per_region_stds))
    print(f"   Median std_rms across {len(per_region_stds)} region(s): {std_rms:.6e}")
    if len(per_region_stds) > 1:
        spread = float(np.std(per_region_stds))
        vals_fmt = ", ".join(f"{x:.6e}" for x in per_region_stds)
        print(f"   Individual values: [{vals_fmt}] ; spread = {spread:.6e}")

    # -------------------------------------------------------------------------
    # 7. Compute local annular backgrounds (one per source that has a bck_).
    # -------------------------------------------------------------------------
    print("\n7. Computing local annular backgrounds (bck_<name> minus <name>)...")
    local_bkg, n_annulus, annulus_masks, annulus_label_image = compute_local_backgrounds(
        data, bck_regions_pix, photometry_masks, sigma=3.0, maxiters=5,
        verbose=verbose)
    n_with_bck = len(local_bkg)
    n_total    = len(photometry_masks)
    print(f"   {n_with_bck}/{n_total} source(s) will get local background subtraction.")

    # Save the annulus label image (1=first source's annulus, 2=second, ...)
    file_prefix = f"{prefix}_" if prefix else ''
    annulus_filename = f'{file_prefix}annulus_masks.fits'
    fits.writeto(annulus_filename, annulus_label_image, header, overwrite=True)
    print(f"   ✓ Annulus label image: {annulus_filename}")
    print(f"     (integer label = order of the source in the region file; "
          f"0 = no annulus)")

    # Scalar summaries used by save_results only for display stretch.
    median_global = float(np.median(list(local_bkg.values()))) if local_bkg else 0.0
    std_global    = std_rms

    # -------------------------------------------------------------------------
    # 8. Perform photometry. Pass `data` (NOT a pre-subtracted image) plus
    #    the local_bkg / n_annulus dicts; the photometry function applies
    #    the local subtraction per source.
    # -------------------------------------------------------------------------
    print("\n8. Performing photometry...")
    bunit = header.get('BUNIT', '').strip()
    print(f"   Image units: {bunit}")

    # Determine telescope: command-line override takes priority over header keyword
    if telescope_override is not None:
        telescop = telescope_override.strip().upper()
        print(f"   Telescope (from --telescope override): {telescop}")
    else:
        telescop = header.get('TELESCOP', '').strip().upper()
        if telescop:
            print(f"   Telescope (from header TELESCOP keyword): {telescop}")
        else:
            # DECaLS/Brick FITS files often omit TELESCOP; try INSTRUME as fallback
            instrume = header.get('INSTRUME', '').strip().upper()
            if 'DECAM' in instrume or 'MOSAIC' in instrume:
                telescop = 'DECAM'
                print(f"   Telescope inferred from INSTRUME='{instrume}': treating as DECALS/DECam")
            elif 'MAGZERO' in header:
                # v12: Euclid MER cutouts (e.g. the COOL-LAMPS / SLICE cutouts) carry MAGZERO but no TELESCOP
                telescop = 'EUCLID'
                print(f"   Telescope inferred from the MAGZERO keyword: treating as EUCLID")
            else:
                print("   ERROR: TELESCOP keyword not found in header and no --telescope override given.")
                print("   Please use --telescope JWST, --telescope HST, or --telescope DECALS.")
                sys.exit(1)

    if verbose:
        print(f"\n   === Pixel Scale Diagnostics ===")
        print(f"   Pixel scale: {pixel_scale_arcsec:.6f} arcsec/pixel")
        print(f"   Pixel area: {pixel_area_sr:.6e} sr/pixel")
        print(f"   For reference:")
        print(f"     - JWST NIRCam: ~0.031-0.063 arcsec/pixel")
        print(f"     - JWST MIRI: ~0.11 arcsec/pixel")
        print(f"   ================================")

    if telescop == 'JWST':
        # Resolve absolute-flux calibration uncertainty
        if cal_frac_override is not None:
            cal_frac = float(cal_frac_override)
            print(f"   Cal-uncertainty override applied: {cal_frac*100:.2f}% of flux")
        else:
            cal_frac = get_nircam_cal_frac(filter_name)
        results = perform_jwst_photometry(data, photometry_masks, bunit,
                                 pixel_area_sr, std_rms,
                                 err_data=err_data,
                                 cal_frac=cal_frac, filter_name=filter_name,
                                 local_bkg=local_bkg, n_annulus=n_annulus,
                                 verbose=verbose)
    elif telescop == 'HST':
        results = perform_hst_photometry(data, photometry_masks, bunit,
                             pixel_area_sr, std_rms, header, err_data=err_data,
                             local_bkg=local_bkg, n_annulus=n_annulus,
                             verbose=verbose)
    elif telescop in ('DECALS', 'DECAM'):
        results = perform_decals_photometry(data, photometry_masks, bunit,
                                pixel_area_sr, std_rms, header,
                                ivar_data=ivar_data,
                                local_bkg=local_bkg, n_annulus=n_annulus,
                                verbose=verbose)
    elif telescop == 'EUCLID':
        cal_frac = float(cal_frac_override) if cal_frac_override is not None else 0.0
        results = perform_euclid_photometry(data, photometry_masks, header, std_rms,
                                zeropoint=zeropoint_override, err_data=err_data,
                                cal_frac=cal_frac,
                                filter_name=filter_name or header.get('EXTNAME'),
                                local_bkg=local_bkg, n_annulus=n_annulus,
                                verbose=verbose)
    else:
        print(f"   ERROR: Telescope '{telescop}' is not supported.")
        print(f"   Supported telescopes: JWST, HST, DECALS (or DECAM), EUCLID")
        sys.exit(1)

    # --- Compute aperture centroids (RA, Dec) ---
    # Mask-pixel barycenter: minimises Sigma d^2 to all aperture pixels.
    print("\n9. Computing aperture centroids (RA, Dec)...")
    centroids = compute_mask_centroids_wcs(photometry_masks, wcs)
    for r in results:
        ra, dec = centroids.get(r['region_name'], (float('nan'), float('nan')))
        r['ra_deg']  = ra
        r['dec_deg'] = dec
        print(f"   {r['region_name']}: RA = {ra:.6f} deg, Dec = {dec:.6f} deg")

    # Save results and create visualizations
    save_results(results, header, data, photometry_masks,
                 median_global, std_global, std_rms, prefix=prefix,
                 local_bkg=local_bkg, n_annulus=n_annulus,
                 annulus_masks=annulus_masks, bunit=bunit)
    
    # Print summary
    print("\n" + "="*50)
    print("CUSTOM APERTURE PHOTOMETRY COMPLETE!")
    print("="*50)
    print(f"Processed {len(photometry_regions)} custom apertures")
    print(f"Local backgrounds applied to: {n_with_bck}/{n_total} sources")
    if local_bkg:
        print(f"Local bkg per pixel (median across sources): {median_global:.6e}")
    print(f"Per-pixel noise std_rms: {std_rms:.6e}")
    print("Results saved in FITS and ASCII formats")
    
    return results


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Custom Aperture Photometry from DS9 Region Files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i image.fits -r regions.reg
  %(prog)s -i image.fits -r regions.reg -e 1 -v
  %(prog)s --image image.fits --regions regions.reg --extension 0 --verbose
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--image', 
                       type=str, 
                       required=True,
                       help='Path to FITS image file')
    
    parser.add_argument('-r', '--regions', 
                       type=str, 
                       required=True,
                       help='Path to DS9 region file')
    
    # Optional arguments
    parser.add_argument('-e', '--extension',
                       type=str,
                       default='0',
                       help='FITS extension for science data, by index or name '
                            '(default: 0 for primary HDU; e.g. VIS or NIR-H for the Euclid cutouts)')
    
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose output with diagnostics')
    
    parser.add_argument('-t', '--telescope',
                       type=str,
                       default=None,
                       choices=['JWST', 'HST', 'DECALS', 'DECAM', 'EUCLID'],
                       help='Override the telescope name (JWST, HST, DECALS/DECAM or EUCLID). '
                            'Use this when the TELESCOP keyword is missing from the header.')

    parser.add_argument('--zeropoint',
                       type=float,
                       default=None,
                       help='EUCLID only: AB zeropoint to use instead of the header MAGZERO.')

    parser.add_argument('-o', '--outdir',
                       type=str,
                       default='.',
                       help='Directory for all outputs (default: current directory).')
    
    parser.add_argument('--prefix',
                       type=str,
                       default='',
                       help='Prefix to prepend to all output filenames '
                            '(e.g. --prefix F390W produces F390W_custom_photometry_results.fits)')

    parser.add_argument('--err-extension',
                       type=str,
                       default='ERR',
                       help="FITS extension (name or integer index) holding the "
                            "per-pixel 1-sigma error map. Default 'ERR' (works for "
                            "JWST i2d products). Pass an integer like '2' to use "
                            "an extension number.")

    parser.add_argument('--no-err-map',
                       action='store_true',
                       help='Disable the ERR-map error contribution entirely '
                            '(only the RMS-region polygon error will be used). '
                            'For DECaLS images, use --ivar instead.')

    parser.add_argument('--ivar',
                       type=str,
                       default=None,
                       metavar='IVAR_FITS',
                       help='Path to a DECaLS inverse-variance map FITS file '
                            '(nanomaggies^{-2} per pixel).  Used only for '
                            '--telescope DECALS/DECAM.  When supplied, the ivar '
                            'map provides the per-pixel noise model; pixels with '
                            'ivar=0 are flagged and excluded from the error sum.')

    parser.add_argument('--cal-uncertainty',
                       type=float,
                       default=None,
                       help='Override the JWST/NIRCam absolute-flux calibration '
                            'uncertainty (as a fraction, e.g. 0.02 for 2%%). '
                            'Default: looked up by filter from the FITS '
                            'FILTER keyword (or the --prefix). Filters not in '
                            'the table get 0%%. Pass 0 to disable.')

    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Resolve the ERR extension argument:
    #   --no-err-map               → None (disabled)
    #   --err-extension <int-str>  → int
    #   --err-extension <name>     → str
    if args.no_err_map:
        err_ext = None
    else:
        try:
            err_ext = int(args.err_extension)
        except (TypeError, ValueError):
            err_ext = args.err_extension

    # v12: extension by index or by name
    try:
        sci_ext = int(args.extension)
    except ValueError:
        sci_ext = args.extension

    # v12: write all outputs to --outdir (inputs resolved to absolute paths first)
    import os
    image_path = os.path.abspath(args.image)
    region_path = os.path.abspath(args.regions)
    ivar_path = os.path.abspath(args.ivar) if args.ivar else None
    os.makedirs(args.outdir, exist_ok=True)
    os.chdir(args.outdir)

    # Run photometry
    results = custom_aperture_photometry(
        image_name=image_path,
        region_file=region_path,
        science_ext=sci_ext,
        verbose=args.verbose,
        telescope_override=args.telescope,
        prefix=args.prefix,
        err_ext=err_ext,
        cal_frac_override=args.cal_uncertainty,
        ivar_name=ivar_path,
        zeropoint_override=args.zeropoint,
    )
