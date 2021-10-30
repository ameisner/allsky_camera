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
import allsky_camera.io as io
import allsky_camera.starcat as starcat
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import allsky_camera.common as common

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

    sc = starcat.StarCat()
    bsc = sc.cat_with_pixel_coords(exp.mjd)

    centroids = util.ac_recentroid(exp.detrended, bsc['x'], bsc['y'])

    assert(len(centroids) == len(bsc))
    assert(np.all(bsc.index == centroids.index))

    bsc = pd.concat([bsc, centroids], axis=1)

    bsc = bsc[bsc['qmaxshift'] == 0] # restrict to good centroids

    assert(len(bsc) > 0)

    r_pix = util.zenith_radius_pix(bsc['x'], bsc['y'])

    bsc = bsc[r_pix <= 500] # factor out 500 special number...

    assert(len(bsc) > 0)

    par = common.ac_params()
    # isolation criterion
    bsc = bsc[bsc['BSC_NEIGHBOR_DEG'] > par['iso_thresh_deg']]

    assert(len(bsc) > 0)

    bsc['min_edge_dist_pix'] = util.min_edge_dist_pix(bsc['xcentroid'],
                                                      bsc['ycentroid'])

    bsc['raw_adu_at_centroid'] = \
        exp.raw_image[np.round(bsc['ycentroid']).astype(int),
                      np.round(bsc['xcentroid']).astype(int)].astype(int)

    bsc['zd_deg'] = 90.0 - bsc['alt_deg']

    skycoords = SkyCoord(bsc['RA']*u.deg, bsc['DEC']*u.deg,
                         frame='icrs')

    bsc['lgal'] = skycoords.galactic.l
    bsc['bgal'] = skycoords.galactic.b

    satur = util.check_saturation(exp.raw_image, bsc['xcentroid'],
                                  bsc['ycentroid'])

    bsc.reset_index(drop=True, inplace=True)

    assert(len(satur) == len(bsc))
    assert(np.all(bsc.index == satur.index))

    bsc = pd.concat([bsc, satur], axis=1)

    bsc = bsc[(bsc['satur_centroid'] == 0) & (bsc['satur_box'] == 0) & \
              (bsc['min_edge_dist_pix'] >= 10)]

    assert(len(bsc) > 0)

    bsc.reset_index(drop=True, inplace=True)

    phot = util.ac_aper_phot(exp.raw_image, bsc['xcentroid'],
                             bsc['ycentroid'])

    bsc = pd.concat([bsc, phot], axis=1)

    bsc = bsc[bsc['flux_adu'] > 0]

    bsc.reset_index(drop=True, inplace=True)

    if write_outputs:
        if not dont_write_detrended:
            io.write_image_level_outputs(exp, outdir)

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
