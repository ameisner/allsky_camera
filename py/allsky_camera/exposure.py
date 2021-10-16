"""
allsky_camera.exposure
======================

A class representing an all-sky camera exposure.
"""

import allsky_camera.util as util

class AC_exposure:
    """
    Object encapsulating the contents of a single all-sky camera exposure

    Parameters
    ----------
        fname_im: str
            full file name of FITS image from which to load the exposure

    """

    def __init__(self, fname_im):
        im, h = util.load_exposure_image(fname_im)

        util.check_image_dimensions(im)

        # image file name
        self.fname_im = fname_im

        # pixel data
        self.raw_image = im

        # image header
        self.header = h

        # exposure time in seconds
        self.time_seconds = util.get_exptime(self.header)

        self.is_detrended = False
