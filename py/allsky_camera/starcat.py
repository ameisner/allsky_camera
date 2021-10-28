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
import copy
import pandas as pd

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

    def cat_with_altaz(self, mjd, horizon_cut=True):
        """
        Return bright star catalog with bonus (alt, az) columns at some MJD.

        Parameters
        ----------
            mjd - float
                MJD (scalar) at which to compute (alt, az) as viewed from KPNO.
            horizon_cut - bool (optional)
                If true, downselect to the catalog rows for stars that are
                above the horizon at the specfied MJD.

        Returns
        -------
            result - pandas.core.frame.DataFrame
                Version of this object's catalog with alt_deg, az_deg, mjd
                columns added.
        Notes
        -----
            Assumes the observer is at KPNO. MJD assumed to be scalar for now.

        """

        print('Calculating bright star catalog altitude/azimuth at ' +
              '{:.5f}'.format(mjd))

        altaz = self.compute_altaz(mjd)

        result = copy.deepcopy(self.catalog)

        result['alt_deg'] = [aa.alt.deg for aa in altaz]
        result['az_deg'] = [aa.az.deg for aa in altaz]
        result['mjd'] = mjd

        if horizon_cut:
            result = result[result['alt_deg'] > 0]
            result.reset_index(drop=True, inplace=True)

        return result

    def cat_with_pixel_coords(self, mjd, horizon_cut=True, in_image_cut=True):
        """
        Return bright star catalog with bonus (x, y) columns at some MJD.

        Parameters
        ----------
            mjd : float
                MJD (scalar) at which to compute (alt, az) as viewed from KPNO.
            horizon_cut : bool (optional)
                If true, downselect to the catalog rows for stars that are
                above the horizon at the specfied MJD.
            in_image_cut : bool (optional)
                If true, downselect to the catalog rows for stars that are
                within the image boundaries. For the MDM camera, there is a
                southern region where stars can be above the horizon but
                fall off of the bottom edge of the image.

        Returns
        -------
            result - pandas.core.frame.DataFrame
                Version of this object's catalog with alt_deg, az_deg, mjd,
                x, y columns added.

        Notes
        -----
            Assumes the observer is at KPNO. MJD assumed to be scalar for now.

        """

        print('Calculating bright star catalog predicted pixel coordinates' +
              ' at ' + '{:.5f}'.format(mjd))

        altaz = self.cat_with_altaz(mjd, horizon_cut=horizon_cut)

        xy = util.altaz_to_xy(altaz['alt_deg'], altaz['az_deg'])

        result = pd.concat([altaz, xy], axis=1)

        if in_image_cut:
            result = result[util.in_image_mask(result['x'],
                                               result['y'])]
            result.reset_index(drop=True, inplace=True)

        return result
