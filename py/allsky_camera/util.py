"""
allsky_camera.util
==================

A collection of all-sky camera related utility functions.
"""

import os
import astropy.io.fits as fits
import allsky_camera.common as common
import allsky_camera.io as io
import allsky_camera.analysis.djs_maskinterp as djs_maskinterp

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

def ac_maskinterp(im):
    """
    Interpolate over static bad pixels.

    Parameters
    ----------
        im : numpy.ndarray
            Image that will have some of its pixels interpolated over.

    Returns
    -------
        result : numpy.ndarray
            Version of the input image with static bad locations
            interpolated over.
    """

    # check that im is type float32

    print('Attempting to interpolate over the static bad pixel mask')

    mask = io.load_static_badpix()

    result = djs_maskinterp.average_bilinear(im, (mask != 0))

    return result

def detrend_ac(exp):
    """
    Driver for pixel-level detrending.

    Parameters
    ----------
        exp : allsky_camera.exposure.AC_exposure
            All-sky camera exposure object.

    Notes
    -----
        Does not return anything, but modifies the state of the input
        all-sky camera exposure object.

    """

    print('Attempting to detrend the raw all-sky camera image')

    assert(not exp.detrended)

    im = exp.raw_image.astype('float32')

    # need to subtract dark current and bias here !!

    # interpolate over static bad pixel mask
    im = ac_maskinterp(im)

    exp.is_detrended = True

    exp.detrended = im

    print('Finished detrending the raw all-sky camera image')