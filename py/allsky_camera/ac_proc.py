#!/usr/bin/env python

"""
allsky_camera.ac_proc
=====================

Main pipeline driver for processing all-sky camera images.
"""

import argparse
from datetime import datetime
import time
import os
from allsky_camera.exposure import AC_exposure
import allsky_camera.util as util

def ac_proc(fname_in, outdir=None, dont_write_detrended=False,
            nmp=None):
    """
    Process one all-sky camera image.

    Parameters
    ----------
        fname_in : str
            Full name of raw all-sky camera image file to process.
        outdir : str, optional
            Full path of output directory. If not specified, outputs are not
            written. Default is None.
        dont_write_detrended : bool, optional
            Set True to skip writing of detrended image as a disk space
            saving measure. Default is False (detrended image written).
        nmp : int, optional
            Number of threads for multiprocessing. Default is None,
            in which case multiprocessing is not used.

    Notes
    -----
        nmp multiprocessing option not yet implemented...

    """

    print('Starting all-sky camera reduction pipeline at: ' +
          str(datetime.utcnow()) + ' UTC')

    t0 = time.time()

    write_outputs = (outdir is not None)

    try:
        print('Running on host: ' + str(os.environ.get('HOSTNAME')))
    except:
        print('Could not retrieve hostname!')

    assert(os.path.exists(fname_in))

    exp = AC_exposure(fname_in)

    # pixel-level detrending
    util.detrend_ac(exp)

    dt = time.time()-t0

    print('all-sky camera reduction pipeline took ' + '{:.2f}'.format(dt) +
          ' seconds')
    print('all-sky camera reduction pipeline completed at: ' +
          str(datetime.utcnow()) + ' UTC')
    
if __name__ == "__main__":
    descr = 'run the all-sky camera reduction pipeline on an exposure'

    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument('fname_in', type=str, nargs=1,
                        help="all-sky camera raw image file name")

    parser.add_argument('--outdir', default=None, type=str,
                        help="directory to write outputs in")

    parser.add_argument('--dont_write_detrended', default=False,
                        action='store_true',
                        help="don't write detrended image")

    parser.add_argument('--multiproc', default=None, type=int,
                        help="number of threads for multiprocessing")

    args = parser.parse_args()

    ac_proc(args.fname_in[0], outdir=args.outdir,
            dont_write_detrended=args.dont_write_detrended,
            nmp=args.multiproc)