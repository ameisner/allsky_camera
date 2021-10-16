"""
allsky_camera.common
==============

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
    
    par = {'nx': 1600,
           'ny': 1200,
           'bitpix': 8,
           'meta_env_var': 'ALLSKY_CAMERA_META',
           'raw_satur_val': 255}

    return par
