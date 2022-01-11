import healpy
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import time

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
            HEALPix map as a 1-dimensional array.

    Notes
    -----
        Use ring-order HEALPix convention.

    """

    npix = 12*(nside**2)
    map = np.zeros(npix, dtype=float)
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
        map += gaussian*transp[i]

    map = map / (wt + (wt == 0).astype(int))

    map[wt < 0.005] = np.nan

    return map

def _test():
    t0 = time.time()
    size = 2500
    lon = np.random.uniform(low=0, high=360, size=size)
    lat = np.random.uniform(low=-90, high=90, size=size)

    transp = np.random.uniform(low=0, high=1, size=size)

    transp[lon < 180] *= 0.5

    map = transpmap(lon, lat, transp)

    dt = time.time()-t0
    print(dt, ' seconds')

    healpy.mollview(map)
    plt.show()