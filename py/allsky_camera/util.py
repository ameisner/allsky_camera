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
import astropy.stats as stats
import numpy as np
import pandas as pd
import copy
from allsky_camera.analysis.djs_photcen import djs_photcen
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
import photutils
from astropy.coordinates import get_moon, EarthLocation, SkyCoord
import astropy.units as u
from astropy.time import Time
import allsky_camera.starcat as starcat
from scipy.ndimage import median_filter
import time
import allsky_camera.analysis.medfilt_parallel as medfilt_parallel
from multiprocessing import Pool
import multiprocessing
from functools import lru_cache
from scipy.spatial import KDTree

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
    Check that raw all-sky camera image dimensions are as expected (image not malformed)

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

def subtract_dark_bias(im):
    """
    Use a non-illuminated region to subtract the sum of dark current and bias.

    Parameters
    ----------
        im : numpy.ndarray
            Image from which to subtract the bias/dark. Should be
            of type float rather than int.
    Returns
    -------
        im : numpy.ndarray
            Modified version of input image with bias/dark subtracted.

    Notes
    -----
        For now just subtract an overall scalar offset; would be nice to explore
        more sophisticated options in the future. Uses sigma clipped mean
        with outlier rejection threshold of 3 sigma.
    """

    print('Attempting to subtract dark/bias level using off region...')

    par = common.ac_params()

    x_l = par['off_region_xmin']
    x_u = par['off_region_xmax']

    y_l = par['off_region_ymin']
    y_u = par['off_region_ymax']

    dark_bias_values = np.ravel(im[y_l:y_u, x_l:x_u])

    mean, median, stddev = stats.sigma_clipped_stats(dark_bias_values,
                                                     sigma=3.0)

    im -= mean

    return im

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

    im = subtract_dark_bias(im)

    # interpolate over static bad pixel mask
    im = ac_maskinterp(im)

    exp.is_detrended = True

    exp.detrended = im

    print('Finished detrending the raw all-sky camera image')

def get_starting_mjd(header):
    """
    Determine MJD of start of all-sky camera exposure based on the raw header.

    Parameters
    ----------
        header : astropy.io.fits.header.Header
            astropy FITS image header ojbect

    Returns
    -------
        mjd : float
            MJD at start of all-sky camera exposure

    Notes
    -----
        Given the large pixel size, this MJD doesn't need to be calculated
        with particularly high precision.

        Uses DATE-OBS header card and assumes that DATE-OBS is the
        local time at KPNO (UTC-7). DATE-OBS expected to be something like
        2020-10-11T21:37:14.72

    """

    card = 'DATE-OBS'
    assert(card in header)

    par = common.ac_params()

    # DATE-OBS is actually UTC-7, whereas pandas.Timestamp assumes UTC
    # so we need to correct for this later
    ts = pd.Timestamp(header['DATE-OBS'])

    mjd = ts.to_julian_date() - par['jd_minus_mjd']

    utc_offs_hours = 7.0
    hours_per_day = 24.0

    # MJD is actually 7 hours later than value returned when using
    # the KPNO local timestamp in place of the UTC timestamp (as was
    # done when instantiating the astropy.Time object)

    mjd += utc_offs_hours/hours_per_day

    return mjd

def zd_deg_to_r_pix(zd_deg):
    """

    Parameters
    ----------
        zd_deg : numpy.ndarray
            Zenith distance in degrees.

    Returns
    -------
        r_pix : numpy.ndarray
            Zenith distance in all-sky camera pixels. Same dimensions as input zd_deg.
    """

    par = common.ac_params()

    r_pix = (zd_deg/90.0)*par['horizon_radius_pix'] + par['coeff2']*(zd_deg**2) + \
            par['coeff3']*(zd_deg**3)

    return r_pix


def altaz_to_xy(alt, az):
    """
    Convert (altitude, azimuth) to predicted detector pixel coordinates.

    Parameters
    ----------
        alt : numpy.ndarray
            Altitude in degrees (apparent or true?).
        az  : numpy.ndarray
            Azimuth in degrees.

    Returns
    -------
        xy : pandas.core.frame.DataFrame
            Table of (x, y) pixel coordinates corresponding to (alt_deg, az_deg)
            Column names are 'X' and 'Y'.
    """

    par = common.ac_params()

    zd = 90.0 - alt # zenith distance in degrees

    r_pix = zd_deg_to_r_pix(zd)

    radeg = 180.0/np.pi

    dx = -1.0*np.sin(az/radeg)*r_pix
    dy = np.cos(az/radeg)*r_pix

    # then rotate dx, dy according to rot_deg, to account for the fact
    # that north is close to, but not precisely aligned with, the +Y axis

    _dx = np.cos(par['rot_deg']/radeg)*dx - np.sin(par['rot_deg']/radeg)*dy
    _dy = np.sin(par['rot_deg']/radeg)*dx + np.cos(par['rot_deg']/radeg)*dy

    xy = pd.DataFrame()

    xy['x'] = _dx + par['x_zenith_pix']
    xy['y'] = _dy + par['y_zenith_pix']

    return xy
    
def min_edge_dist_pix(x, y):
    """
    Compute minimum distance to any image edge


    Parameters
    ----------
        x : numpy.ndarray
            x pixel coordinates. Needs to have the same size as y.
        y : numpy.ndarray
            y pixel coordinates. Needs to have the same size as x.

    Returns
    -------
        min_edge_dist : numpy.ndarray
            For each (x, y) pair, the minimum distance to any image edge.

    Notes
    -----
        Works for array-valued x, y. Edge is taken to be outer edge of
        the boundary pixel, with a convention that the center of a pixel
        has integer (x, y) coordinates and zero-indexed indexing.

    """

    min_edge_dist = 20000

    par = common.ac_params()

    min_edge_dist = np.minimum(x + 0.5, y + 0.5)
    min_edge_dist = np.minimum(min_edge_dist, par['nx'] - 0.5 - x)
    min_edge_dist = np.minimum(min_edge_dist, par['ny'] - 0.5 - y)

    return min_edge_dist

def in_image_mask(x, y):
    """
    Return a boolean mask indicating which (x, y) pairs fall within image.

    Parameters
    ----------
        x : numpy.ndarray
            x pixel coordinates
        y : numpy.ndarray
            y pixel coordinates

    Returns
    -------
        in_image : numpy.ndarray
            Boolean mask with same dimensions as input x (and y). True means
            within image boundaries, false means outside of image boundaries

    Notes
    -----
        In my convention, a 2D numpy array representing an image is indexed as
        [y, x].

    """

    min_edge_dist = min_edge_dist_pix(x, y)

    mask = min_edge_dist > 0

    return mask

def recentroid_1chunk(im, x, y):
    """"
    Refine a list of centroids relative to initial guess.

    Parameters
    ----------
        im : numpy.ndarray
            Detrended all-sky camera image. Should be floating point, though
            this is also ensured within this function itself.

        x : numpy.ndarray
            List of x pixel values for initial centroid guesses.

        y : numpy.ndarray
            List of y pixel values for initial centroid guesses.

    Returns
    -------
        result : pandas.core.frame.DataFrame
            Columns include xcentroid, ycentroid - the refine centroids.

    Notes
    -----
        x and y need to have the same length. In my convention, a 2D numpy
        array representing an image is indexed as [y, x].

    """

    n = len(x)

    xcen = np.zeros(n, dtype=float)
    ycen = np.zeros(n, dtype=float)
    qmaxshift = np.zeros(n, dtype=int)

    cmaxshift = 2.0

    for i in range(n):
        _xcen, _ycen, q = djs_photcen(x[i], y[i], im, cbox=4, cmaxiter=10,
                                      cmaxshift=cmaxshift, ceps=0)

        xcen[i] = _xcen
        ycen[i] = _ycen
        qmaxshift[i] = q

    result = pd.DataFrame()

    result['xcentroid'] = xcen
    result['ycentroid'] = ycen
    result['x_shift'] = xcen - x
    result['y_shift'] = ycen - y
    result['qmaxshift'] = qmaxshift

    result['centroid_shift_flag'] = (np.abs(result['x_shift']) > cmaxshift) | (np.abs(result['y_shift']) > cmaxshift) | (qmaxshift != 0)

    return result

def ac_recentroid(_im, x, y, nmp=None):
    """"
    Driver for refining centroids relative to initial guess.

    Parameters
    ----------
        _im : numpy.ndarray
            Detrended all-sky camera image. Should be floating point, though
            this is also ensured within this function itself.

        x : numpy.ndarray
            List of x pixel values for initial centroid guesses.

        y : numpy.ndarray
            List of y pixel values for initial centroid guesses.

        nmp : int (optional)
            Number of threads for multiprocessing. Default is None,
            which means that multiprocessing will not be used.

    Returns
    -------
        result : pandas.core.frame.DataFrame
            Columns include xcentroid, ycentroid - the refine centroids.

    Notes
    -----
        x and y need to have the same length. In my convention, a 2D numpy
        array representing an image is indexed as [y, x].

        It appears that nmp=2 may actually be slower than no multiprocessing
        due to serialization overhead (may depend on the machine being used).

    """

    t0 = time.time()
    print('Refining bright star centroids')

    im = _im.astype('float32')

    assert(len(x) == len(y))

    if (nmp is None) or (nmp == 1):
        result = recentroid_1chunk(im, x, y)
    else:
        assert (nmp <= multiprocessing.cpu_count())
        print('Running parallelized recentroiding...')
        p = Pool(nmp)
        xy = pd.DataFrame()
        xy['x'] = x
        xy['y'] = y
        dfs = np.array_split(xy, nmp)
        args = [(im, np.array(_df['x']), np.array(_df['y'])) for _df in dfs]
        results = p.starmap(recentroid_1chunk, args)
        result = pd.concat(results, axis=0)
        result.reset_index(drop=True, inplace=True)
        p.close()
        p.join()

    dt = time.time()-t0
    print('Centroid refinement took ' + '{:.2f}'.format(dt) + ' seconds')
    return result

def zenith_radius_pix(x_pix, y_pix):
    """
    Compute distance in pixels of a detector location relative to zenith.

    Parameters
    ----------
        x_pix : numpy.ndarray
            x pixel locations. Needs to have the same dimensions as y_pix.
        y_pix : numpy.ndarray
            y pixel locations. Needs to have the same dimensions as x_pix.

    Returns
    -------
        radius_pix : numpy.ndarray
            Radius in pixels of the (x_pix, y_pix) locations on the 
            detector relative to the nominal zenith.

    """

    assert(x_pix.shape == y_pix.shape)

    par = common.ac_params()

    dx_pix = x_pix - par['x_zenith_pix']
    dy_pix = y_pix - par['y_zenith_pix']

    radius_pix = np.hypot(dx_pix, dy_pix)

    return radius_pix

def check_saturation(raw, x, y):
    """
    Compute saturation flags for a list of star centroid locations.

    Parameters
    ----------
        raw : numpy.ndarray
            Raw all-sky camera image.
        x   : numpy.ndarray
            List of star centroid x coordinates. Needs to have the same
            length as y.
        y   : numpy.ndarray
            List of star centroid y coordinates. Needs to have the same
            length as x.

    Returns
    -------
        flags : pandas.core.frame.DataFrame
            Dataframe with columns related to labeling saturation at/near
            each (x, y) centroid location. Number of rows equal to lenght of
            x and y.

    Notes
    -----
        The image passed in needs to be the truly raw image, not the
        detrended image.
    """

    print('Computing saturation flags...')

    assert(len(x) == len(y))
    assert(len(x) > 0)
    assert(len(raw.shape) == 2)

    assert(np.sum(np.round(raw) != raw) == 0)

    n = len(x)
    x = np.array(x)
    y = np.array(y)

    par = common.ac_params()

    ix = np.round(x).astype(int)
    # + 1 accounts for Python indexing convention
    ix_upper = np.minimum(np.maximum(ix + 1, 0), par['nx'] - 1).astype(int) + 1
    ix_lower = np.minimum(np.maximum(ix - 1, 0), par['nx'] - 1).astype(int)

    iy = np.round(y).astype(int)
    iy_upper = np.minimum(np.maximum(iy + 1, 0), par['ny'] - 1).astype(int) + 1
    iy_lower = np.minimum(np.maximum(iy - 1, 0), par['ny'] - 1).astype(int)

    bad_centroid = np.zeros(n, dtype=int)

    bad_centroid += (raw[iy, ix] == 255).astype(int)
    bad_centroid += 2*(raw[iy, ix] >= 240).astype(int)

    bad_box = np.zeros(n, dtype=int)

    for i in range(n):
        box = raw[iy_lower[i]:iy_upper[i], ix_lower[i]:ix_upper[i]]
        boxmax = np.max(box)
        bad_box[i] += (boxmax == 255).astype(int)
        bad_box[i] += 2*((boxmax >= 240).astype(int))

    df = pd.DataFrame()

    df['satur_centroid'] = bad_centroid 
    df['satur_box'] = bad_box

    return df

def _get_area_from_ap(ap):
    """
    Retrieve the area of an aperture photometry aperture.

    Parameters
    ----------
        ap : photutils.aperture.circle.CircularAperture
            Photutils aperture object (don't believe that it technically
            needs to be circular though).

    Returns
    -------
        area : float
            Area of the aperture in image pixels.

    Notes
    -----
        This utility exists to try and work around the photutils API change
        between versions 0.6 and 0.7.

    """

    if (photutils.__version__.find('0.7') != -1) or (photutils.__version__.find('1.0') != -1) or (photutils.__version__.find('1.2') != -1) or (photutils.__version__.find('1.3') != -1):
        area = ap.area # 0.7
    else:
        area = ap.area() # 0.6

    return area

def ac_aper_phot(_im, x, y, bg_sigclip=False):
    """
    Perform aperture photometry at a list of star centroid locations.

    Parameters
    ----------
        _im : numpy.ndarray
            2D array containing the all-sky camera image. Should be the
            detrended image, not the raw image.
        x   : numpy.ndarray
            List of star centroid x coordinates. Needs to have the same
            length as y.
        y   : numpy.ndarray
            List of star centroid y coordinates. Needs to have the same
            length as x.
        bg_sigclip : bool (optional)
            Do sigma clipping when computing annulus background levels.
            Default is False, which is faster.

    Returns
    -------
        cat : pandas.core.frame.DataFrame
            Dataframe encapsulating the photometry results.
    """

    print('Attempting to perform aperture photometry...')
    t0 = time.time()

    assert(len(x) == len(y))

    x = np.array(x)
    y = np.array(y)

    im = _im.astype(float)

    par = common.ac_params()

    positions = list(zip(x, y))

    cat = pd.DataFrame() # initialize output

    # will I ever use more than one aperture radius?

    radii = [par['aper_phot_objrad']]
    ann_radii = par['annulus_radii'] # should have 2 elements - inner and outer

    apertures = [CircularAperture(positions, r=r) for r in radii]
    annulus_apertures = CircularAnnulus(positions, r_in=ann_radii[0],
                                        r_out=ann_radii[1])
    annulus_masks = annulus_apertures.to_mask(method='center')

    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(im)
        annulus_data_1d = annulus_data[mask.data > 0]
        if bg_sigclip:
            # this sigma_clipped_stats call is actually the slow part !!
            _, median_sigclip, std_bg = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
        else:
            bkg_median.append(np.median(annulus_data_1d))

    bkg_median = np.array(bkg_median)
    phot = aperture_photometry(im, apertures)

    for i, aperture in enumerate(apertures):
        aper_bkg_tot = bkg_median*_get_area_from_ap(aperture)
        cat['aper_sum_bkgsub_' + str(i)] = phot['aperture_sum_' + str(i)] - aper_bkg_tot

        cat['aper_bkg_' + str(i)] = aper_bkg_tot

    cat['sky_annulus_area_pix'] = _get_area_from_ap(annulus_apertures)
    cat['sky_annulus_median'] = bkg_median

    flux_adu = np.zeros((len(cat), len(radii)), dtype=float)

    for i in range(len(radii)):
        flux_adu[:, i] = cat['aper_sum_bkgsub_' + str(i)]

    cat['flux_adu'] = flux_adu

    dt = time.time()-t0
    print('Done with aperture photometry...took ' + '{:.2f}'.format(dt) + ' seconds')
    return cat

def get_moon_position(mjd):
    """
    Get Moon (ra, dec).

    Parameters
    ----------
        mjd : float
            MJD at which to compute the Moon's coordinates.

    Returns
    -------
        coords : astropy.coordinates.sky_coordinate.SkyCoord
            SkyCoord object providing the Moon's (RA, Dec) at the desired MJD.

    Notes
    -----
        For now, intended for scalar MJD input, not array-valued.

    """

    par = common.ac_params()

    location = EarthLocation(lat=par['kpno_lat']*u.deg,
                             lon=par['kpno_lon']*u.deg,
                             height=par['kpno_elev_meters']*u.m)

    time = Time(str(mjd), format='mjd')

    coords = get_moon(time, location=location)

    return coords

def trim_catalog_moon(cat, mjd):
    """
    Trim star catalog to avoid including sky locations near the Moon.

    Parameters
    ----------
        cat : pandas.core.dataframe.DataFrame
            Bright star catalog.
        mjd : float
            MJD at which to compute the Moon's coordinates.

    Returns
    -------
        trim : pandas.core.dataframe.DataFrame
            Copy of the input catalog with rows within 15 degrees of the Moon
            removed.

    """

    print('Cutting sources potentially affected by the Moon...')

    moon = get_moon_position(mjd)

    trim = copy.deepcopy(cat)

    stars = SkyCoord(cat['RA']*u.deg, cat['DEC']*u.deg)

    ang_sep = moon.separation(stars)

    thresh = 15.0 # degrees of Moon separation; factor out to ac_params?

    trim['moon_sep_deg'] = np.array(ang_sep, dtype=float)

    trim = trim[ang_sep.deg > thresh]

    # add printout regarding how many sources were rejected due to
    # Moon proximity?
    print('Removed ' + str(len(cat)-len(trim)) + ' sources nearby the Moon...')

    trim.reset_index(drop=True, inplace=True)

    return trim

def catalog_galactic_coords(cat):
    """
    Compute galactic coordinates for catalog and add these as new columns.

    Parameters
    ----------
        cat : pandas.core.dataframe.DataFrame
            Catalog for which to compute Galactic coordinates. Needs to
            have columns 'RA' and 'DEC' (with values in degrees)

    Returns
    -------
        cat : pandas.core.dataframe.DataFrame
            Modified version of input catalog with Galactic coordinates
            added in columns named 'lgal' and 'bgal' (units of degrees).
    """

    skycoords = SkyCoord(cat['RA']*u.deg, cat['DEC']*u.deg,
                         frame='icrs')

    cat['lgal'] = np.array(skycoords.galactic.l, dtype=float)
    cat['bgal'] = np.array(skycoords.galactic.b, dtype=float)

    return cat

def ac_phot(exp, cat):
    """
    Wrapper for aperture photometry and some related catalog repackaging.

    Parameters
    ----------
        exp : allsky_camera.exposure.AC_exposure
            All-sky camera exposure object for the image being analyzed.
        cat : allsky_camera.starcat.StarCat
            Bright star catalog object. Should already have refined
            centroids computed and stored in the 'xcentroid' and
            'ycentroid' columns.

    Returns
    -------
        cat : allsky_camera.starcat.StarCat
            Modified version of input catalog with added photometry columns,
            and also trimmed to sources found to have positive aperture flux.
    """

    phot = ac_aper_phot(exp.detrended, cat['xcentroid'], cat['ycentroid'])

    cat = pd.concat([cat, phot], axis=1)

    cat = cat[cat['flux_adu'] > 0]

    cat.reset_index(drop=True, inplace=True)

    cat['m_inst'] = -2.5 * np.log10(cat['flux_adu'] / exp.time_seconds)

    return cat

def ac_catalog(exp, nmp=None, force_mp_centroiding=False):
    """
    Driver to perform bright star catalog recentroiding and aperture photometry.

    Parameters
    ----------
        exp : allsky_camera.exposure.AC_exposure
            All-sky camera exposure object for the image being analyzed.

        nmp : int (optional)
            Number of threads for multiprocessing. Default is None,
            which means that multiprocessing will not be used. Gets passed
            on to the recentroiding step.
        force_mp_centroiding : bool, optional
            When multiprocessing has been requested via nmp > 1,
            this boolean dictates whether to use multiprocessing
            for recentroiding. Default is False, as there are
            indications that (at least for nmp=2), the
            image serialization overhead makes parallelization
            of centroid refinement a net loss.

    Returns
    -------
        bsc : pandas.core.dataframe.DataFrame
            Bright star catalog including refined centroids and
            aperture photometry.

    """

    sc = starcat.StarCat()
    bsc = sc.cat_with_pixel_coords(exp.mjd)

    centroids = ac_recentroid(exp.detrended, bsc['x'], bsc['y'],
                              nmp=(nmp if force_mp_centroiding else None))

    assert(len(centroids) == len(bsc))
    assert(np.all(bsc.index == centroids.index))

    bsc = pd.concat([bsc, centroids], axis=1)

    bsc = bsc[bsc['qmaxshift'] == 0] # restrict to good centroids

    assert(len(bsc) > 0)

    print('Attempting to flag wrong centroids...')
    t0 = time.time()
    bsc = flag_wrong_centroids_kdtree(bsc)
    dt = time.time()-t0
    print('flagging wrong centroids took ', '{:.3f}'.format(dt), ' seconds')

    r_pix = zenith_radius_pix(bsc['x'], bsc['y'])

    par = common.ac_params()

    bsc = bsc[r_pix <= par['r_pix_safe']]

    assert(len(bsc) > 0)

    # isolation criterion
    bsc = bsc[bsc['BSC_NEIGHBOR_DEG'] > par['iso_thresh_deg']]

    assert(len(bsc) > 0)

    bsc['min_edge_dist_pix'] = min_edge_dist_pix(bsc['xcentroid'],
                                                 bsc['ycentroid'])

    bsc['zd_deg'] = 90.0 - bsc['alt_deg']

    bsc = catalog_galactic_coords(bsc)

    bsc['raw_adu_at_centroid'] = \
        exp.raw_image[np.round(bsc['ycentroid']).astype(int),
                      np.round(bsc['xcentroid']).astype(int)].astype(int)

    satur = check_saturation(exp.raw_image, bsc['xcentroid'],
                                  bsc['ycentroid'])

    bsc.reset_index(drop=True, inplace=True)

    assert(len(satur) == len(bsc))
    assert(np.all(bsc.index == satur.index))

    bsc = pd.concat([bsc, satur], axis=1)

    bsc = bsc[(bsc['satur_centroid'] == 0) & (bsc['satur_box'] == 0) & \
              (bsc['min_edge_dist_pix'] >= 10)]

    assert(len(bsc) > 0)

    bsc.reset_index(drop=True, inplace=True)

    bsc = ac_phot(exp, bsc)

    bsc = trim_catalog_moon(bsc, exp.mjd)

    return bsc

def r_pix_to_zd(r_pix):
    """
    Convert from zenith distance in pixels to zenith distance in degrees.

    Parameters
    ----------
        r_pix : numpy.ndarray
            Zenith distance in pixels.

    Returns
    -------
        zd_deg : numpy.ndarray
            Zenith distance in degrees.

    Notes
    -----
        Need to think more about apparent versus true zenith distance...
    """

    par = common.ac_params()

    zd_deg = np.polyval(par['icoeff'], r_pix)

    zd_deg = np.maximum(zd_deg, 0)

    return zd_deg

def pixel_solid_angle(zd_deg):
    """
    Compute spatially varying pixel solid angle in square arcminutes.

    Parameters
    ----------
        zd_deg : numpy.ndarray
            Zenith distance in degrees.

    Returns
    -------
        area_sq_arcmin : numpy.ndarray
            Pixel solid angle in units of square arcminutes.

    """

    r_pix = zd_deg_to_r_pix(zd_deg)

    par = common.ac_params()

    # dr/d(z_d) derivative
    dr_dzd = par['horizon_radius_pix']/90.0 + 2*par['coeff2']*zd_deg + 3*par['coeff3']*(zd_deg**2)

    area = (zd_deg/r_pix)*(1.0/dr_dzd) # sq deg

    # are these special handling steps for being at/near zenith necessary?
    area_zenith = (90.0/par['horizon_radius_pix'])**2 # sq deg
    area[r_pix <= 0] = area_zenith

    area_sq_arcmin = area*(60.0**2) # square arcminutes

    return area_sq_arcmin

def sky_brightness_map(detrended, exptime, nmp=None):
    """
    Make a map of sky magnitude per square arcsecond.

    Parameters
    ----------
        detrended : numpy.ndarray
            2D numpy array representing the detrended all-sky camera image.
        exptime : float
            Exposure time in seconds.
        nmp: int
            Number of threads for multiprocessing.

    Returns
    -------
        mag_per_sq_asec : numpy.ndarray
            2D numpy array representing the sky brightness map.
    """

    print('Making a map of the sky brightness in V mag per square arcsec...')

    sh = detrended.shape

    assert(len(sh) == 2)

    par = common.ac_params()

    ybox, xbox = np.mgrid[0:par['ny'], 0:par['nx']]

    r_pix = zenith_radius_pix(xbox, ybox)

    zd_deg = r_pix_to_zd(r_pix)

    zd_deg[r_pix > par['r_pix_safe']] = np.nan

    n_rows_good = np.sum(r_pix <= par['r_pix_safe'], axis=0)

    good_col_indices = np.where(n_rows_good > 0)

    npix_pad = (par['ksize'] // 2) + 1
    ind_l = np.min(good_col_indices) - npix_pad
    ind_u = np.max(good_col_indices) + 1 + npix_pad

    ind_l = max(ind_l, 0)
    ind_u = min(ind_u, par['nx'])

    detrended_subimage = detrended[:, ind_l:ind_u]

    if (nmp is None) or (nmp == 1):
        print('Computing median filtered version of the detrended image...')
        t0 = time.time()
        med_subimage = median_filter(detrended_subimage, par['ksize'])
        dt = time.time() - t0
        print('Done computing median filtered image...took ' + '{:.2f}'.format(dt) + ' seconds')
    else:
        med_subimage = medfilt_parallel.split_and_reassemble(detrended_subimage,
                                                    nchunks=nmp, ksize=par['ksize'],
                                                    nmp=nmp)

    med = np.zeros((par['ny'], par['nx']), dtype=float) + np.nan

    med[:, ind_l:ind_u] = med_subimage

    sq_arcmin = pixel_solid_angle(zd_deg)

    # put sky counts in ADU / sec / pix
    med = med / exptime

    # put sky counts in ADU / sec / (sq arcmin)
    med = med / sq_arcmin

    # put sky counts in ADU / sec / (sq asec)
    med = med / 3600.0

    mag_per_sq_asec = -2.5 * np.log10(med) + par['zp_adu_per_s']

    return mag_per_sq_asec

@lru_cache(maxsize=2)
def circular_mask(radius_pix):
    """
    Create a boolean mask representing a circle centered at the image center.
    Parameters
    ----------
        radius_pix : float
            Radius value to use for the circular mask.
    Returns
    -------
        mask : numpy.ndarray
            Boolean mask as a 2D numpy array. True means that a pixel
            location is within radius_pix pixels of the center.
    Notes
    -----
        Possible speed-ups?
        Generalize so that there can be a user-specified central coordinates?
        Make radius_pix into an optional keyword arg that defaults to the size
        of the on-sky image region?

    """

    par = common.ac_params()

    sh = (par['ny'], par['nx'])

    Y, X = np.ogrid[:par['ny'], :par['nx']]

    x_center = par['x_zenith_pix']
    y_center = par['y_zenith_pix']

    dist = np.hypot(X - x_center, Y - y_center)

    dist = dist.reshape(sh)

    mask = (dist <= radius_pix)

    return mask

def flag_wrong_centroids(cat, full_cat):
    """
    Flag cases of a centroid wandering off to an entirely different star.

    Parameters
    ----------
        cat : pandas.core.dataframe.DataFrame
            Table with columns including xcentroid, ycentroid, MY_BSC_ID. Can be a
            subset of the rows for the entire pointing camera exposure's
            catalog, with the idea being that partial lists can be run
            in parallel to reduce run time.
        full_cat : pandas.core.dataframe.DataFrame
            Table with columns including x_gaia_guess, y_gaia_guess, SOURCE_ID.
            Needs to be the full star catalog for this pointing camera exposure.
    Returns
    -------
        cat : pandas.core.dataframe.DataFrame
            Input catalog cat but with an added column called wrong_source_centroid,
            which has a value of 1 (centroid has wandered too far) or 0.
    """

    wrong_source_centroid = np.zeros(len(cat), dtype=bool)

    for i in range(len(cat)):
        _dist = np.sqrt(np.power(full_cat['x'] - cat.iloc[i]['xcentroid'], 2) + \
                        np.power(full_cat['y'] - cat.iloc[i]['ycentroid'], 2))
        indmin = np.argmin(_dist)
        wrong_source_centroid[i] = (full_cat.iloc[indmin]['MY_BSC_ID'] != cat.iloc[i]['MY_BSC_ID'])

    cat['wrong_source_centroid'] = wrong_source_centroid.astype(int)

    return cat

def flag_wrong_centroids_kdtree(cat):
    """
    Flag cases of a centroid wandering off to an entirely different star.

    Parameters
    ----------
        cat : pandas.core.dataframe.DataFrame
            Table with columns including xcentroid, ycentroid, MY_BSC_ID.

    Returns
    -------
        cat : pandas.core.dataframe.DataFrame
            Input catalog cat but with an added column called wrong_source_centroid,
            which has a value of 1 (centroid has wandered too far) or 0.

    Notes
    -----
        The idea is that using a KDTree should make this fast...

    """

    n = len(cat)

    tree = KDTree(np.c_[cat['x'], cat['y']])

    dists, inds = tree.query(np.array((cat['xcentroid'],cat['ycentroid'])).T, k=1)

    inds = inds.reshape(n)

    cat['wrong_source_centroid'] = (inds != np.arange(n)).astype(int)

    return cat