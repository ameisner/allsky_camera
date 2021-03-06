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
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pkg_resources import resource_filename
import copy
from functools import lru_cache
from scipy.stats import scoreatpercentile
import allsky_camera.transpmap as transpmap
import pickle

@lru_cache()
def load_static_badpix():
    """
    Read in static bad pixel mask.

    Returns
    -------
        mask : numpy.ndarray
            Static bad pixel mask. 1 means bad, 0 means good.
    """
    par = common.ac_params()

    fname = resource_filename('allsky_camera', os.path.join('data', par['static_mask_filename']))

    print('READING ' + fname)

    assert(os.path.exists(fname))

    mask = fits.getdata(fname)

    return mask

@lru_cache()
def load_bsc():
    """
    Load BSC5 catalog.

    Returns
    -------
        df : pandas.core.frame.DataFrame
            BSC5 catalog as a pandas dataframe, sorted by Dec (low to high)
    """

    par = common.ac_params()

    fname = resource_filename('allsky_camera', os.path.join('data', par['bsc_filename_csv']))

    print('READING ' + fname)

    assert(os.path.exists(fname))

    df = pd.read_csv(fname)

    # sort by Dec in case this comes in useful for binary searching later on
    df.sort_values('DEC', inplace=True, ignore_index=True)

    return df

def write_detrended(exp, outdir):
    """
    Write the detrended all-sky camera image.

    Parameters
    ----------
        exp    : allsky_camera.exposure.AC_exposure
                 All-sky camera exposure object.
        outdir : str
                 Full path of output directory.

    """

    print('Attempting to write detrended image output file')

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

def write_sbmap(exp, sbmap, outdir):
    """
    Write the sky brightness map as a FITS image file.

    Parameters
    ----------
        exp    : allsky_camera.exposure.AC_exposure
                 All-sky camera exposure object.
        sbmap  : numpy.ndarray
                 2D image of the sky brightness.
        outdir : str
                 Full path of output directory.

    """

    print('Attempting to write surface brightness map image output')

    assert(os.path.exists(outdir))

    outname = (os.path.split(exp.fname_im))[-1]

    outname = outname.replace('.fits', '-sbmap.fits')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    header = exp.header
    header['BUNIT'] = 'V mag per sq asec'

    hdu = fits.PrimaryHDU(sbmap.astype('float32'), header=header)
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
            have columns 'x', 'y', 'xcentroid', 'ycentroid'
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

def sky_brightness_plot(sbmap, exp, outdir):
    """
    Plot a map of the sky brightness with a corresponding color bar.

    Parameters
    ----------
        sbmap : numpy.ndarray
            2D image representing the sky brightness as measured by the
            all-sky camera. Contains NaN values denoting detector locations at
            excessively low elevation.
        exp : allsky_camera.exposure.AC_exposure
                 All-sky camera exposure object.
        outdir : str
                 Full path of output directory.
    """

    plt.cla()

    assert(os.path.exists(outdir))

    basename = (os.path.split(exp.fname_im))[-1]

    title = basename + ' ; altitude > 20 deg'
    plt.title(title)

    vmin = np.nanmin(sbmap)
    vmax = np.nanmax(sbmap)

    sbmap_masked = np.ma.masked_where(np.logical_not(np.isfinite(sbmap)), sbmap)

    cmap = copy.copy(matplotlib.cm.get_cmap("gray_r"))
    cmap.set_bad('#b5d1ff', 1.)

    ims = plt.imshow(sbmap_masked, vmin=vmin, vmax=vmax, interpolation='nearest',
                     origin='lower', cmap=cmap)

    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(ims, cax=cax)

    cbar.ax.set_ylabel('V mag per sq asec', fontsize=16)

    outname = basename.replace('.fits', '-sbmap.png')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    plt.savefig(outname_tmp, bbox_inches='tight', format='png')

    plt.cla()
    plt.close()

    os.rename(outname_tmp, outname)

def oplot_centroids(cat, exp, outdir):
    """
    Overplot source catalog centroids on detrended image.

    Parameters
    ----------
        cat : pandas.core.dataframe.DataFrame
            Bright star catalog with all-sky camera photometry. Needs to
            have columns 'x', 'y', 'xcentroid', 'ycentroid'
        exp : allsky_camera.exposure.AC_exposure
            All-sky camera exposure object.
        outdir : str
            Full path of output directory.

    """

    plt.cla()

    par = common.ac_params()

    import allsky_camera.util as util
    mask = util.circular_mask(par['r_pix_safe'])

    limits = scoreatpercentile(np.ravel(exp.detrended[mask]), [1, 97])
    vmin = limits[0]
    vmax = limits[1]

    plt.imshow(exp.detrended, vmin=0, vmax=255, origin='lower',
               interpolation='nearest', cmap='gray')

    plt.scatter(cat['xcentroid'], cat['ycentroid'], s=5, edgecolor='r',
                facecolor='none', linewidth=0.1)

    plt.xticks([])
    plt.yticks([])

    basename = (os.path.split(exp.fname_im))[-1]

    title = basename + ' ; refined centroids overplotted'
    plt.title(title)

    outname = basename.replace('.fits', '-detrended.png')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    plt.savefig(outname_tmp, bbox_inches='tight', format='png', dpi=400)

    plt.cla()
    plt.close()

    os.rename(outname_tmp, outname)

def _add_healpix_header_cards(h, nside, extname=None, coordsys=None):
    """
    Add standard HEALPix header cards to a FITS header.

    Parameters
    ----------
        h : astropy.io.fits.header.Header
            FITS header object. Will be modified via the addition of
            additional header cards and their corresponding comments.
        nside : int
            HEALPix Nside value that will be added to the header.
        extname : str, optional
            Extension name. If None (default) then no EXTNAME keyword will
            be added.
        coordsys : str, optional
            Coordinate system (either EQU for equatorial or HOR for
            horizontal). Default is None. If specified, will go into the
            COORDSYS header card.

    Notes
    -----
        Example of what this function is aiming for:

        PIXTYPE = 'HEALPIX '           / Pixel algorithm
        ORDERING= 'RING    '           / Ordering scheme
        NSIDE   =                    8 / Resolution parameter
        NPIX    =                  768 / # of pixels
        FIRSTPIX=                    0 / First pixel (0 based)
        LASTPIX =                  767 / Last pixel (0 based)

        Should probably add something like BUNIT to specify the units of
        each extension...

    """

    npix = 12*nside*nside

    h['PIXTYPE'] = ('HEALPIX', 'Pixel algorithm')
    h['ORDERING'] = ('RING', 'Ordering scheme')
    h['NSIDE'] = (nside, 'Resolution parameter')
    h['NPIX'] = (npix, '# of pixels')
    h['FIRSTPIX'] = (0, 'First pixel (0 based)')
    h['LASTPIX'] = (npix-1, 'Last pixel (0 based)')

    if extname is not None:
        h['EXTNAME'] = extname

    if coordsys is not None:
        h['COORDSYS'] = (coordsys, 'HEALPix coordinate system')

def write_healpix(exp, cat, outdir, nside=8):
    """
    Write transparency/zeropoint related HEALPix maps.

    Parameters
    ----------
        exp : allsky_camera.exposure.AC_exposure
            All-sky camera exposure object.
        cat : pandas.core.dataframe.DataFrame
            Bright star catalog with all-sky camera photometry. Needs to
            have columns 'x', 'y', 'xcentroid', 'ycentroid'
        outdir : str
            Full path of output directory.
        nside : int, optional
            HEALPix Nside parameter. Needs to be a valid parameter of 2. The
            Thinking is that Nside=8 or Nside=4 would be the best choices.

    Notes
    -----
        Add checks on HEALPix Nside provided?

    """

    print('Making HEALPix maps...')

    par = common.ac_params()

    zp_map_hor, zp_counts_hor = transpmap.healmap_median(cat['az_deg'],
                                                         cat['alt_deg'],
                                                         cat['zp_adu_per_s'],
                                                         nside=nside)

    zp_map_equ, zp_counts_equ = transpmap.healmap_median(cat['RA'],
                                                         cat['DEC'],
                                                         cat['zp_adu_per_s'],
                                                         nside=nside)

    transp_map_hor = np.power(10, -(par['zp_adu_per_s'] - zp_map_hor)/2.5)
    transp_map_equ = np.power(10, -(par['zp_adu_per_s'] - zp_map_equ)/2.5)


    hdu_zp_hor = fits.PrimaryHDU(data=zp_map_hor)
    hdu_counts_hor = fits.ImageHDU(data=zp_counts_hor)

    hdu_zp_equ = fits.ImageHDU(data=zp_map_equ)
    hdu_counts_equ = fits.ImageHDU(data=zp_counts_equ)

    hdu_transp_hor = fits.ImageHDU(data=transp_map_hor)
    hdu_transp_equ = fits.ImageHDU(data=transp_map_equ)

    hdul = [hdu_zp_hor, hdu_counts_hor, hdu_zp_equ, hdu_counts_equ,
            hdu_transp_hor, hdu_transp_equ]
    extnames = ['ZP_HOR', 'N_ZP_HOR', 'ZP_EQU', 'N_ZP_EQU',
                'TRANSP_HOR', 'TRANSP_EQU']

    for hdu, extname in zip(hdul, extnames):
        coordsys = 'EQU' if 'EQU' in extname else 'HOR'
        _add_healpix_header_cards(hdu.header, nside, extname=extname,
                                  coordsys=coordsys)

    hdul = fits.HDUList(hdul)

    basename = (os.path.split(exp.fname_im))[-1]

    outname = basename.replace('.fits', '-healpix.fits')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    hdul.writeto(outname_tmp)

    os.rename(outname_tmp, outname)

def write_streaks(exp, streaks, outdir):
    """
    Write a file with a summary of detected streaks.

    Parameters
    ----------
        exp : allsky_camera.exposure.AC_exposure
            All-sky camera exposure object.
        streaks : list
            List of streaks, each of which is a dictionary with data
            defining one detected streak.
        outdir : str
            Full path of output directory.
    Notes
    -----
        If input streaks variable is an empty list (no streaks detected) then
        no output is written.

        Streaks list contains numpy arrays as dictionary values, which
        makes it not possible to dump to JSON, hence the pickle output file
        type.

    """

    if not len(streaks):
        print('Streaks file not written because no streaks were cataloged.')
        return

    assert(os.path.exists(outdir))

    outname = (os.path.split(exp.fname_im))[-1]

    outname = outname.replace('.fits', '-streaks.pkl')

    outname = os.path.join(outdir, outname)

    outname_tmp = outname + '.tmp'

    assert(not os.path.exists(outname))
    assert(not os.path.exists(outname_tmp))

    print('Writing satellite streaks...')

    pickle.dump(streaks, open(outname_tmp, "wb" ) )

    os.rename(outname_tmp, outname)
