"""
allsky_camera.starcat
======================

A class representing a bright star catalog.
"""

import astropy.io.fits as fits
import allsky_camera.util as util # needed?
import allsky_camera.io as io
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
import allsky_camera.common as common

class StarCat:
    """
    Object encapsulating a bright star catalog.

    Notes
    -----
        Maybe should have some attribute tracking which bright star catalog
        is being used, in the event that we have multiple options in the
        future?

    """
    def __init__(self):
        self.catalog = io.load_bsc() # double check the function name here
        
    def compute_altaz(self, mjd):
        """
        Compute (altitude, azimuth) coords of the catalog at specified MJD.

        Parameters
        ----------
            mjd - float
                MJD (scalar) at which to compute (alt, az) as viewed from KPNO.

        Returns
        -------
            altaz - astropy.coordinates.sky_coordinate.SkyCoord
                List of (alt, az) pairs in degrees. Same length as
                the bright star catalog.

        Notes
        -----
            Assumes the observer is at KPNO. MJD assumed to be scalar for now.

        """

        par = common.ac_params()

        obstime = Time(str(mjd), format='mjd')

        location = EarthLocation(lat=par['kpno_lat']*u.deg,
                                 lon=par['kpno_lon']*u.deg,
                                 height=par['kpno_elev_meters']*u.m)

        coords = SkyCoord(self.catalog['RA']*u.deg, self.catalog['DEC']*u.deg)

        # this is slow ...
        # pressure is needed to account for atmospheric refraction,
        # to match default behavior of IDL eq2hor
        altaz = coords.transform_to(AltAz(obstime=obstime, location=location,
                                          pressure=par['pressure_bars']*u.bar))

        return altaz

    def cat_with_altaz(self, mjd):
        """
        Return bright star catalog with bonus (alt, az) columns.

        Parameters
        ----------
            mjd - float
                MJD (scalar) at which to compute (alt, az) as viewed from KPNO.

        Notes
        -----
            Assumes the observer is at KPNO. MJD assumed to be scalar for now.

        """
        pass
