"""
allsky_camera.common
====================

Central location for parameters expected to be used in
various other places throughout the codebase.

Goal is to factor out special numbers.
"""

def ac_params():
    """
    Provide a dictionary of various special numbers and other parameters

    Returns:
        dictionary of parameters

    Notes:
        The values here apply to the MDM all-sky camera sample FITS readouts 
        from 2020 October. In the future this function be generalized via
        keyword argument or similar to handle multiple different all-sky
        camera setups.

    """

    # may want to include astrometric solution parameters here as well

    # camera-specific parameters
    par_camera = {'nx': 1600,
                  'ny': 1200,
                  'bitpix': 8,
                  'raw_satur_val': 255,
                  'exptime_card': 'EXPOSURE',
                  'static_mask_filename': 'static_badpix_mask.fits.gz',
                  'off_region_xmin': 1450,
                  'off_region_xmax': 1600,
                  'off_region_ymin': 1075,
                  'off_region_ymax': 1200,
                  'x_zenith_pix' :  802.37891,
                  'y_zenith_pix' : 511.35899,
                  'rot_deg' : 3.5506619,
                  'coeff2' : 0.0048194290,
                  'coeff3' : -4.3659004e-5,
                  'horizon_radius_pix' : 626.05372,
                  'aper_phot_objrad' : 2.0,
                  'annulus_radii' : [5.0, 10.0]}

    # miscellaneous parameters not tied to a particular camera
    par_misc = {'meta_env_var': 'ALLSKY_CAMERA_META',
                'jd_minus_mjd' : 2400000.5,
                'bsc_filename' : 'bsc5.fits',
                'bsc_filename_csv' : 'bsc5.csv',
                'kpno_lon' : -111.5997,
                'kpno_lat' : 31.9599,
                'kpno_elev_meters' : 2100.0,
                'pressure_bars' : 1.013,
                'iso_thresh_deg': 0.3}

    return dict(par_camera, **par_misc)
