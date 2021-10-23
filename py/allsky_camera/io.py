"""
allsky_camera.io
================

I/O functions for all-sky camera reduction pipeline.
"""

import astropy.io.fits as fits
import allsky_camera.common as common
import os
import pandas as pd

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

def load_bsc():
    """
    Load BSC5 catalog.

    Returns
    -------
        df : pandas.core.frame.DataFrame
            BSC5 catalog as a pandas dataframe, sorted by Dec (low to high)
    """

    par = common.ac_params()

    fname = os.path.join(os.environ[par['meta_env_var']],
                     par['bsc_filename_csv'])

    assert(os.path.exists(fname))

    df = pd.read_csv(fname)

    # sort by Dec in case this comes in useful for binary searching later on
    df.sort_values('DEC', inplace=True, ignore_index=True)

    return df

def write_image_level_outputs(exp, outdir):
    """
    Write image level outputs, such as the detrended all-sky camera image.

    Parameters
    ----------
        exp    : allsky_camera.exposure.AC_exposure
                 All-sky camera exposure object.
        outdir : str
                 Full path of output directory.

    Notes
    -----
        Currently this only writes out a detrended image. Perhaps in the future
        there could also be an image-level bitmask written out.
    """

    print('Attempting to write image level outputs')

    assert(os.path.exists(outdir))

    outname = (os.path.split(exp.fname_im))[-1]

    outname = outname.replace('.fits', '-detrended.fits')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    hdu = fits.PrimaryHDU(exp.detrended.astype('float32'), header=exp.header)
    hdu.writeto(outname_tmp)
    os.rename(outname_tmp, outname)
