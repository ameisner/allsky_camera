"""
allsky_camera.io
================

I/O functions for all-sky camera reduction pipeline.
"""

import astropy.io.fits as fits
import allsky_camera.common as common
import os

def load_static_badpix():
    """
    Read in static bad pixel mask.

    Returns
    -------
        mask : numpy.ndarray
            Static bad pixel mask. 1 means bad, 0 means good.
    """
    par = common.ac_params()

    fname = os.path.join(os.environ[par['meta_env_var']],
                         par['static_mask_filename'])

    assert(os.path.exists(fname))

    mask = fits.getdata(fname)

    return mask
