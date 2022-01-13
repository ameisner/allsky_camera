import healpy
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import time
from multiprocessing import Pool
import pandas as pd

def healmap_median(lon, lat, vals, nside=8):
    """
    Median by HEALPix pixel for a list of values at a list of sky locations.

    Parameters
    ----------
        lon : numpy.ndarray
            List of longitude data points, in degrees. Must have same
            number of elements as lat.
        lat : numpy.ndarray
            List of latitude data points, in degrees. Must have same
            number of elements as lon.
        vals : numpy.ndarray
            List of values (e.g., transparency, zeropoint, ...) at (lon, lat). Must have same
            number of elements as lon and lat arrays. Values assumed to be of type float.
        nside : int, optional
            HEALPix nside parameter, default is nside=8.

    Returns
    -------
        medians : numpy.ndarray
            HEALPix median map as a 1-dimensional array. NaN for missing data
            (e.g., regions that are below the horizon).
        counts : numpy.ndarray
             HEALPix map of the number of stars per HEALPix pixel. 0 means no stars...

    Notes
    -----
        Use ring-order HEALPix convention.

    """

    npix = 12*nside*nside

    ipix = healpy.ang2pix(nside, lon, lat, lonlat=True, nest=False)

    df = pd.DataFrame()
    df['ipix'] = ipix
    df['vals'] = vals

    df_med = df.groupby(['ipix'])['vals'].agg(['median', 'count'])

    counts = np.zeros(npix, dtype=int)
    medians = np.full(npix, np.nan, dtype=float)

    counts[df_med.index] = df_med['count']
    medians[df_med.index] = df_med['median']

    return medians, counts

def transpmap(lon, lat, transp, nside=32):
    """
    Build transparency map from a list of points and their transparencies.

    Parameters
    ----------
        lon : numpy.ndarray
            List of longitude data points, in degrees. Must have same
            number of elements as lat.
        lat : numpy.ndarray
            List of latitude data points, in degrees. Must have same
            number of elements as lon.
        transp : numpy.ndarray
            List of transparency values at (lon, lat). Must have same
            number of elements as lon and lat arrays.
        nside : int, optional
            HEALPix nside parameter, default is nside=32 (1.83 deg pixels).

    Returns
    -------
        map : numpy.ndarray
            HEALPix transparency map as a 1-dimensional array. NaN for missing data
            (e.g., regions that are below the horizon).
        tot : numpy.ndarray
             HEALPix map of the accumulated sum of transparency measurements
             (intermediate product).
        wt : numpy.ndarray
             HEALPix weight map proportional to number of stars effectively contributing
             to each pixel.

    Notes
    -----
        Use ring-order HEALPix convention.

    """

    npix = 12*(nside**2)
    map = np.zeros(npix, dtype=float)
    tot = np.zeros(npix, dtype=float)
    wt = np.zeros(npix, dtype=float)
    npoints = len(lon)

    assert(len(lon) == len(lat))
    assert(len(lon) == len(transp))

    # smoothing kernel size
    fwhm_deg = 7.0
    sigma_deg = fwhm_deg/2.355

    ipix = np.arange(npix)
    nest = False
    heal_lon, heal_lat = healpy.pixelfunc.pix2ang(nside, ipix, nest=nest, lonlat=True)

    sc_grid = SkyCoord(heal_lon*u.deg, heal_lat*u.deg)

    for i in range(npoints):
        print(i)
        # compute angular distance (in degrees) of each healpix pixel
        sc_star = SkyCoord(lon[i]*u.deg, lat[i]*u.deg)
        # evaluate gaussian at those angular distances centered at (lon[i], lat[i])
        # be careful about radians vs deg !!!!

        angsep_deg = sc_star.separation(sc_grid).deg

        gaussian = np.exp(-1.0*(angsep_deg**2)/(2*(sigma_deg**2)))
        gaussian /= np.sum(gaussian) # normalize...

        wt += gaussian
        tot += gaussian*transp[i]

    map = tot / (wt + (wt == 0).astype(int))

    map[wt < 0.005] = np.nan

    return map, tot, wt

def transpmap_parallel(lon, lat, transp, nside=32, nmp=None):
    """
    Build transparency map from a list of points and their transparencies.

    Parameters
    ----------
        lon : numpy.ndarray
            List of longitude data points, in degrees. Must have same
            number of elements as lat.
        lat : numpy.ndarray
            List of latitude data points, in degrees. Must have same
            number of elements as lon.
        transp : numpy.ndarray
            List of transparency values at (lon, lat). Must have same
            number of elements as lon and lat arrays.
        nside : int, optional
            HEALPix nside parameter, default is nside=32 (1.83 deg pixels).
        nmp : int, optional
            Number of multiprocessing processes to use. Default of None means
            that no multiprocessing will be used.

    Returns
    -------
        map : numpy.ndarray
            HEALPix transparency map as a 1-dimensional array. NaN for missing data
            (e.g., regions that are below the horizon).
        tot : numpy.ndarray
             HEALPix map of the accumulated sum of transparency measurements
             (intermediate product).
        wt : numpy.ndarray
             HEALPix weight map proportional to number of stars effectively contributing
             to each pixel.

    Notes
    -----
        Use ring-order HEALPix convention.
        Basically a wrapper for the transpmap function.

    """

    t0 = time.time()
    if (nmp == 1) or (nmp is None):
        return transpmap(lon, lat, transp, nside=nside)
    else:
        p = Pool(nmp)
        lons = np.array_split(lon, nmp)
        lats = np.array_split(lat, nmp)
        transps = np.array_split(transp, nmp)
        args = [(lons[i], lats[i], transps[i], nside) for i in range(nmp)]
        results = p.starmap(transpmap, args)

        tot = results[0][1]
        wt = results[0][2]
        for i in range(1, nmp):
            tot += results[i][1]
            wt += results[i][2]

        map = tot / (wt + (wt == 0).astype(int))

        map[wt < 0.005] = np.nan

        p.close()
        p.join()

    dt = time.time()-t0

    return map, tot, wt

def _test(nmp=None):
    t0 = time.time()
    size = 2500
    lon = np.random.uniform(low=0, high=360, size=size)
    lat = np.random.uniform(low=-90, high=90, size=size)

    transp = np.random.uniform(low=0, high=1, size=size)

    transp[lon < 180] *= 0.5

    #map, tot, wt = transpmap(lon, lat, transp)

    map, tot, wt = transpmap_parallel(lon, lat, transp, nside=16, nmp=nmp)

    dt = time.time()-t0
    print(dt, ' seconds')

    healpy.mollview(map)
    plt.show()

def _test_median(nside=8):

    size = 2500

    lon = np.random.uniform(low=0, high=360, size=size)
    lat = np.random.uniform(low=-90, high=90, size=size)
    transp = np.random.uniform(low=0, high=1, size=size)

    transp[lon < 180] *= 0.5

    medians, counts = healmap_median(lon, lat, transp, nside=nside)

    healpy.mollview(counts)
    plt.show()

    healpy.mollview(medians)
    plt.show()

    return medians
