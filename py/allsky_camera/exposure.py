"""
allsky_camera.exposure
======================

A class representing an all-sky camera exposure.
"""

import allsky_camera.util as util
import allsky_camera.common as common
import os
import astropy.io.fits as fits
import copy

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

        self.detrended = None

        self.mjd = util.get_starting_mjd(self.header)

        self.streaks = None

    def _tmp_detrended_filename(self):
        """
        Get name of temporary detrended file.

        Returns
        -------
            outname : str
                Full temporary file name.

        """

        outname = os.path.split(self.fname_im)[-1].replace('.fits',
            '-detrended.fits.tmp')

        outname = os.path.join('/tmp', outname)

        return outname

    def _write_tmp_detrended(self, null_edge=False):
        """
        Write detrended image and header to /tmp as a FITS image file.

        Parameters
        ----------
            null_edge : bool, optional
                Null out edge pixel values (the one edge-most row/column
                along each boundary) so that astride will 'close' the
                contours of satellite streaks that extend off the image edges.
                Could imagine trying other choices, like some value less
                than the minimum pixel value across the entire detrended image.
                Or maybe NaN values? Could also try nulling N > 1 edge-most
                rows/columns, rather than just N = 1.
        Notes
        -----
            This is a workaround to accommodate the astride Streak
            constructor.
        """

        outname = self._tmp_detrended_filename()

        if null_edge:
            par = common.ac_params()
            im = copy.deepcopy(self.detrended)
            im[:, 0] = 0
            im[:, par['nx']-1] = 0
            im[0, :] = 0
            im[par['ny']-1, :] = 0
            hdu = fits.PrimaryHDU(im, self.header)
        else:
            hdu = fits.PrimaryHDU(self.detrended, self.header)

        hdu.writeto(outname)

    def _del_tmp_detrended(self):
        """
        Delete temporary detrended image FITS file.

        """

        fname = self._tmp_detrended_filename()

        if os.path.exists(fname):
            os.remove(fname)
