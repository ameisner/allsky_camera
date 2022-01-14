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
import multiprocessing

if ('HOSTNAME' in os.environ) and ('cori' in os.environ['HOSTNAME']):
    try:
        import desiutil.iers
        desiutil.iers.freeze_iers()
    except:
        print('desiutil not available?')

def ac_proc(fname_in, outdir=None, dont_write_detrended=False,
            nmp=None, skip_checkplots=False, skip_sbmap=False,
            write_sbmap=False, force_mp_centroiding=False,
            dont_write_catalog=False, oplot_centroids=False,
            write_healpix=False):
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
        skip_checkplots : bool, optional
            Set True to avoid writing any PNG checkplot output files.
            Default is False.
        skip_sbmap : bool, optional
            Set True to skip computing sky brightness map.
            Default is False.
        write_sbmap : bool, optional
            Set True to write FITS image file with sky brightness map.
            Default is False.
        force_mp_centroiding : bool, optional
            When multiprocessing has been requested via nmp > 1,
            this boolean dictates whether to use multiprocessing
            for recentroiding. Default is False, as there are
            indications that (at least for nmp=2), the
            image serialization overhead makes parallelization
            of centroid refinement a net loss.
        dont_write_catalog : bool, optional
            Set True to skip writing of all-sky camera source catalog.
            Default is False.
        oplot_centroids : bool, optional
            Set True to make and write out a checkplot overlaying bright
            star centroids on the detrended image.
        write_healpix : bool, optional
            Set True to write a low-res HEALPix map of the photometric
            zeropoint, ultimately for use by DESI survey planning.

    Notes
    -----
        nmp multiprocessing option only partially implemented...

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

    if nmp is not None:
        assert(nmp <= multiprocessing.cpu_count())

    exp = AC_exposure(fname_in)

    # pixel-level detrending
    util.detrend_ac(exp)

    bsc = util.ac_catalog(exp, nmp=nmp, force_mp_centroiding=force_mp_centroiding)

    if not skip_sbmap:
        sbmap = util.sky_brightness_map(exp.detrended, exp.time_seconds, nmp=nmp)

    if write_outputs:
        # skip making output dir if all outputs have individually
        # been switched off?
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if not dont_write_catalog:
             io.write_source_catalog(bsc, exp, outdir)

        if not dont_write_detrended:
            io.write_detrended(exp, outdir)

        if write_sbmap:
            io.write_sbmap(exp, sbmap, outdir)

        if write_healpix:
            io.write_healpix(exp, bsc, outdir)
            
        if not skip_checkplots:
            io.zp_checkplot(bsc, exp, outdir)
            io.centroid_quiver_plot(bsc, exp, outdir)
            if not skip_sbmap:
                io.sky_brightness_plot(sbmap, exp, outdir)
            if oplot_centroids:
                io.oplot_centroids(bsc, exp, outdir)

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

    parser.add_argument('--skip_checkplots', default=False,
                        action='store_true',
                        help="don't create checkplots")

    parser.add_argument('--skip_sbmap', default=False,
                        action='store_true',
                        help="don't do sky brightness map")

    parser.add_argument('--write_sbmap', default=False,
                        action='store_true',
                        help="write sky brightness map as FITS image")

    parser.add_argument('--force_mp_centroiding', default=False,
                        action='store_true',
                        help="use multiprocessing for recentroiding")

    parser.add_argument('--dont_write_catalog', default=False,
                        action='store_true',
                        help="don't write source catalog FITS file")

    parser.add_argument('--oplot_centroids', default=False,
                        action='store_true',
                        help="checkplot overlaying centroids on detrended image")
    parser.add_argument('--write_healpix', default=False,
                        action='store_true',
                        help="write HEALPix map of photometric zeropoint")

    args = parser.parse_args()

    ac_proc(args.fname_in[0], outdir=args.outdir,
            dont_write_detrended=args.dont_write_detrended,
            nmp=args.multiproc, skip_checkplots=args.skip_checkplots,
            skip_sbmap=args.skip_sbmap, write_sbmap=args.write_sbmap,
            force_mp_centroiding=args.force_mp_centroiding,
            dont_write_catalog=args.dont_write_catalog,
            oplot_centroids=args.oplot_centroids,
            write_healpix=args.write_healpix)
