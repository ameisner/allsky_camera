"""
allsky_camera.util
==================

A collection of all-sky camera related utility functions.
"""

import os
import astropy.io.fits as fits

def load_exposure_image(fname):
    """
    Read in FITS image and its header for one all-sky camera exposure.

    Parameters
    ----------
        fname : str

    Returns
    -------
        im : numpy.ndarray
             the image pixel data as a two-dimensional numpy array
        h  : astropy.io.fits.header.Header
             the image's corresponding astropy FITS header

    """

    assert(os.path.exists(fname))

    print('Attempting to read exposure : ' + fname)

    im, h = fits.getdata(fname, header=True)

    print('Successfully read raw pointing camera image file')

    return im, h
