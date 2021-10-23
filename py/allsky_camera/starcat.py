"""
allsky_camera.starcat
======================

A class representing a bright star catalog.
"""

import astropy.io.fits as fits
import allsky_camera.util as util # needed?
import allsky_camera.io as io

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

        Notes
        -----
            Assumes the observer is at KPNO. MJD assumed to be scalar for now.

        """
        pass

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
