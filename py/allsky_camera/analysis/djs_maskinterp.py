"""
allsky_camera.analysis.djs_maskinterp
=====================================

Meant to be a Python port of djs_maskinterp.pro from IDLUTILS.
"""

import numpy as np
from scipy.interpolate import interp1d
import astropy.io.fits as fits
import time

def maskinterp1(yval, mask):
# omitting xval arg (assume regular grid), omitting const kw arg
# (assume const=True behavior is desired)

    yval = np.array(yval)

    mask = mask.astype(int)

    bad = (mask != 0)
    if np.sum(bad) == 0:
        return yval

    good = (mask == 0)
    ngood = np.sum(good)
    if ngood == 0:
        return yval
    
    if np.sum(good) == 1:
        return yval*0 + yval[good][0]

    ynew = yval
    ny = len(yval)

    igood = (np.where(good))[0]
    ibad = (np.where(bad))[0]
    f = interp1d(igood, yval[igood], kind='linear', fill_value='extrapolate')

    yval[bad] = f(ibad)

    # do the /const part
    if igood[0] != 0:
        ynew[0:igood[0]] = ynew[igood[0]]
    if igood[ngood-1] != (ny-1):
        ynew[(igood[ngood-1]+1):ny] = ynew[igood[ngood-1]]

    return ynew

def maskinterp(yval, mask, axis):
    """
    Linearly interpolate along one axis over masked pixel locations.

    Parameters
    ----------
        yval : numpy.ndarray
            2D image, pixel values should be float not int data type
        mask : numpy.ndarray
            2D mask image, should have same dimensions as yval, should have
            an integer data type.
            Nonzero values mark pixels that will be interpolated over.
        axis : int
            Axis along which to interpolate, should be either 0 or 1

    Returns
    -------
        yval : numpy.ndarray
            Version of input yval 2D image where pixel locations with
            nonzero mask values have been interpolated over along one
            dimension.
    """

    mask = mask.astype(int)

    sh_yval = yval.shape
    sh_mask = mask.shape

    assert(len(sh_yval) == 2)
    assert(len(sh_mask) == 2)

    assert((sh_yval[0] == sh_mask[0]) and (sh_yval[1] == sh_mask[1]))

    assert((axis == 0) or (axis == 1))

    wbad = (np.where(mask != 0))

    if axis == 0:
        # the y coord values of rows that need some interpolation
        bad_stripe_indices = np.unique(wbad[0])
    else:
        # the x coord values of columns that need some interpolation
        bad_stripe_indices = np.unique(wbad[1])

    if len(bad_stripe_indices) == 0:
        return yval

    for ind in bad_stripe_indices:
        if axis == 0:
            yval[ind, :] = maskinterp1(yval[ind, :], mask[ind, :])
        else:
            yval[:, ind] = maskinterp1(yval[:, ind], mask[:, ind])

    return yval

def average_bilinear(yval, mask):
    """
    Interpolate separately along the x and y directions, then take the mean.

    Parameters
    ----------
        yval : np.ndarray
            2D image, pixel values should be float not int data type
        mask : np.ndarray
            2D mask image, should have same dimensions as yval, should have
            an integer data type.
            Nonzero values mark pixels that will be interpolated over.

    Returns
    -------
        interp : np.ndarray
            2D image, same dimensions as yval and mask, should be the
            same as yval except for pixel locations with nonzero values in
            mask, which have been interpolated over

    Notes
    -----
        Meant to be the equivalent of the following common pattern
        using djs_maskinterp.pro in IDL:

            intx = djs_maskinterp(yval, mask, iaxis=0, /const)
            inty = djs_maskinterp(yval, mask, iaxis=1, /const)

            interp = (intx + inty)/2.0
    """

    int0 = maskinterp(yval, mask, 0)
    int1 = maskinterp(yval, mask, 1)
    interp = (int0 + int1)/2.0

    return interp
