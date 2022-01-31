"""
allsky_camera.satellites
==========================

Satellite streak detection.
"""

from astride import Streak
import time
import os

def detect_streaks(exp):
    """
    Run streak detection.

    Parameters
    ----------
        exp : allsky_camera.exposure.AC_exposure
            All-sky camera exposure object with detrended image available.

    Returns
    -------
        streaks : list
            List of streaks, each of which is a dictionary with data
            defining one detected streak.

    Notes
    -----
        A lot more work could be done tuning the astride parameters
        that presumably exist.

        astride discards contours that aren't closed, which is a problem
        for long all-sky camera exposures where the streaks often extend
        off of one or more image boundaries.

        Streak detection should operate on the detrended (rather than raw)
        all-sky camera image.

    """

    print('Running streak detection...')

    t0 = time.time()

    exp._write_tmp_detrended(null_edge=True)

    fname = exp._tmp_detrended_filename()

    streak = Streak(fname)

    streak.detect()

    exp._del_tmp_detrended()

    streaks = streak.streaks

    dt = time.time() - t0

    print('Found ' + str(len(streaks)) + ' streaks')
    print('Streak detection took ' + '{:.2f}'.format(dt) + ' seconds ; ' +
          os.path.basename(exp.fname_im))

    exp.streaks = streaks

    return streaks
