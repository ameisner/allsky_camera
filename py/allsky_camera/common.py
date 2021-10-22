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

    par = {'nx': 1600,
           'ny': 1200,
           'bitpix': 8,
           'meta_env_var': 'ALLSKY_CAMERA_META',
           'raw_satur_val': 255,
           'exptime_card': 'EXPOSURE',
           'static_mask_filename': 'static_badpix_mask.fits.gz',
           'off_region_xmin': 1450,
           'off_region_xmax': 1600,
           'off_region_ymin': 1075,
           'off_region_ymax': 1200,
           'jd_minus_mjd' : 2400000.5}

    return par
