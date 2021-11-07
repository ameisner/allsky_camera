"""
allsky_camera.io
================

I/O functions for all-sky camera reduction pipeline.
"""

import astropy.io.fits as fits
import allsky_camera.common as common
import os
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

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

def write_source_catalog(catalog, exp, outdir):
    """
    Write the catalog of bright star position/flux measurements.

    Parameters
    ----------
        catalog : pandas.core.frame.DataFrame
                  The star catalog that will be written out.
        exp     : allsky_camera.exposure.AC_exposure
                  All-sky camera exposure object.
        outdir  : str
                  Full path of output directory.

    """

    print('Attempting to write catalog output')

    assert(os.path.exists(outdir))

    outname = (os.path.split(exp.fname_im))[-1]

    outname = outname.replace('.fits', '-cat.fits')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    tab = Table.from_pandas(catalog)

    hdul = []
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.BinTableHDU(data=tab, name='CATALOG'))

    hdul = fits.HDUList(hdul)

    hdul.writeto(outname_tmp)

    os.rename(outname_tmp, outname)

def zp_checkplot(cat, exp, outdir):
    """
    Make/write a checkplot showing the all-sky camera versus BSC mags.

    Parameters
    ----------
        cat : pandas.core.dataframe.DataFrame
            Bright star catalog with all-sky camera photometry. Needs to
            have columns 'm_inst' and 'VMAG'
        exp : allsky_camera.exposure.AC_exposure
                 All-sky camera exposure object.
        outdir : str
                 Full path of output directory.

    """

    plt.cla()

    xtitle = 'BSC V magnitude (' + str(len(cat)) + ' stars)'
    ytitle = '-2.5' + r'$\times$' + 'log'  + r'$_{10}$' + '(ADU/sec)'

    plt.scatter(cat['VMAG'], cat['m_inst'], s=10, edgecolor='none')

    # use sigma-clipped median instead?
    zp = -1.0*np.median(cat['m_inst'] - cat['VMAG'])

    xmin = np.min(cat['VMAG'])
    xmax = np.max(cat['VMAG'])

    xsamp = np.array([xmin, xmax])
    ysamp = xsamp - zp

    plt.plot(xsamp, ysamp, c='r')

    pad_mags = 0.04
    xlim = (xmin-pad_mags, xmax+pad_mags)

    plt.xlim(xlim)

    ylim = (xlim[0] - 1 - zp, xlim[1] + 1 - zp)
    plt.ylim(ylim)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    print('Attempting to write photometric zeropoint checkplot...')

    assert(os.path.exists(outdir))

    basename = (os.path.split(exp.fname_im))[-1]

    outname = basename.replace('.fits', '-zp.png')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    title = basename

    # will this have a problem if somehow zp is NaN-like?
    title += ' ; ZP = ' + '{:.3f}'.format(zp) + ' mag'

    plt.title(title)
    plt.savefig(outname_tmp, dpi=200, bbox_inches='tight', format='png')

    plt.cla()

    os.rename(outname_tmp, outname)

def centroid_quiver_plot(cat, exp, outdir):
    """
    Make/write a checkplot showing a quiver plot of the astrometric residuals.

    Parameters
    ----------
        cat : pandas.core.dataframe.DataFrame
            Bright star catalog with all-sky camera photometry. Needs to
            have columns 'dx', 'dy', 'xcentroid', 'ycentroid'
        exp : allsky_camera.exposure.AC_exposure
                 All-sky camera exposure object.
        outdir : str
                 Full path of output directory.

    """

    plt.cla()
    dx = cat['xcentroid'] - cat['x']
    dy = cat['ycentroid'] - cat['y']

    plt.figure(figsize=(8, 8))

    q = plt.quiver(cat['x'], cat['y'], dx, dy, scale=10, scale_units='inches')

    # would be nice to add an arrow/legend indicating what the arrow size means
    # (what the units are)

    title = os.path.split(exp.fname_im)[-1]

    # where does the altitude > 20 cut happen?
    plt.title(title + ' ; astrometric model residuals ; altitude > 20 deg')

    plt.xlabel('x pixel coordinate', fontsize=12)
    plt.ylabel('y pixel coordinate', fontsize=12)

    assert(os.path.exists(outdir))

    basename = (os.path.split(exp.fname_im))[-1]

    outname = basename.replace('.fits', '-quiver.png')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    plt.savefig(outname_tmp, bbox_inches='tight', format='png')

    plt.cla()

    os.rename(outname_tmp, outname)