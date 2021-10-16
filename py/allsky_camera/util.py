"""
allsky_camera.util
==================

A collection of all-sky camera related utility functions.
"""

import os
import astropy.io.fits as fits
import allsky_camera.common as common

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

    print('Successfully read raw all-sky camera image file')

    return im, h

def get_exptime(h_im):
    """
    Retrieve exposure time in seconds.

    Parameters
    ----------
        h_im : astropy.io.fits.header.Header
            all-sky camera image header

    Returns
    -------
        exptime_s : float
            exposure time in seconds

    Notes
    -----
        Right now this function is pretty trivial but could take on
        some nuances down the road.
    """

    par = common.ac_params()

    card = par['exptime_card']

    if card in h_im:
        exptime_s = h_im[card]
        print('Exposure time comment from header: ' + h_im.comments[card])
    else:
        print('Could not find an exposure time in the raw image header!')
        # maybe in the future could substitute in some
        # 'default' exposure time here...
        assert(False)

    print('Exposure time is ' + '{:.2f}'.format(exptime_s) + ' seconds')

    return float(exptime_s)

def check_image_dimensions(image):
    """
    Check that raw all-sky camera image are as expected (image not malformed)

    Parameters
    ----------
        image : numpy.ndarray
            all-sky camera image

    Notes
    -----
        Doesn't matter if image raw or detrended (at least for the case of the
        sample MDM all-sky camera FITS readouts.

    """

    print('Checking raw all-sky camera image dimensions...')

    par = common.ac_params()

    sh = image.shape

    assert(sh[0] == par['ny'])
    assert(sh[1] == par['nx'])

    print('Raw all-sky image has correct dimensions')
