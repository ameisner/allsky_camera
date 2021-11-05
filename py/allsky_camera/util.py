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

    r_pix = (zd/90.0)*par['horizon_radius_pix'] + par['coeff2']*(zd**2) + \
            par['coeff3']*(zd**3)

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

def ac_recentroid(_im, x, y):
    """"
    Refine centroids relative to initial guess.

    Parameters
    ----------
        _im : numpy.ndarray
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

    print('Refining bright star centroids')

    im = _im.astype(float)

    assert(len(x) == len(y))

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

    radius_pix = np.sqrt(dx_pix**2 + dy_pix**2)

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

    if (photutils.__version__.find('0.7') != -1) or (photutils.__version__.find('1.0') != -1):
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

    trim['moon_sep_deg'] = ang_sep

    trim = trim[ang_sep.deg > thresh]

    # add printout regarding how many sources were rejected due to
    # Moon proximity?
    print('Removed ' + str(len(cat)-len(trim)) + ' sources nearby the Moon...')

    trim.reset_index(drop=True, inplace=True)

    return trim
