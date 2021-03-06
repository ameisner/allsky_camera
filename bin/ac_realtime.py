#!/usr/bin/env python

import argparse
import glob
import time
import os
import allsky_camera.ac_proc as pipeline

files_processed = []

def _reduce_new_files(flist_fits, outdir='.'):
    """
    Run reductions of new raw all-sky camera images.

    Parameters
    ----------
        flist_fits : list
            List of file names (full paths) for which to run the reduction pipeline.
        outdir : str (optional)
            Directory in which to write outputs.

    """

    for f_fits in flist_fits:

        if not os.path.exists(f_fits):
            print('FITS file disappeared?? skipping')
            continue

        # call the reduction pipeline
        print('Reducing ' + f_fits)

        try:
            pipeline.ac_proc(f_fits, outdir=outdir, dont_write_detrended=True,
                             skip_checkplots=False)
        except:
            print('PROCESSING FAILURE: ' + f_fits)

def detect_new_files(data_dir, outdir='.'):
    """
    Detect and keep track of files that have not been previously processed.

    Parameters
    ----------
        data_dir : str
            Full path of raw data directory.
        outdir : str (optional)
            Directory in which to write outputs.

    """

    print('Checking for new .fits files...')

    assert(os.path.exists(data_dir))

    global files_processed

    flist_fits = glob.glob(data_dir + '/*.fits')

    if len(flist_fits) == 0:
        print('No data to reduce yet...')
        return

    flist_fits_new = set(flist_fits) - set(files_processed)

    if len(flist_fits_new) > 0:
        flist_fits_new = list(flist_fits_new)
        flist_fits_new.sort()

        _reduce_new_files(flist_fits_new, outdir=outdir)

        files_processed = files_processed + flist_fits_new

def _watch(data_dir, wait_seconds=5, outdir='.'):
    """
    Watch raw data directory for new FITS image files to process.

    Parameters
    ----------
        data_dir :  str
            Name of directory to watch for new FITS image files.
        wait_seconds : float (optional)
            Polling interval in seconds (default is 5 seconds).
        outdir : str (optional)
            Directory in which to write the outputs.

    """

    while True:
        print('Waiting', wait_seconds, ' seconds')
        time.sleep(wait_seconds)
        detect_new_files(data_dir, outdir=outdir)

def _do_veto(fname):
    """
    Veto any further processing of a list of previously processed files.

    Parameters
    ----------
        fname : str
            File name with list of files to exclude from processing.
            Should be an ASCII file with one file name (full path) per line.

    """
    assert(os.path.exists(fname))

    global files_processed

    f = open(fname, 'r')

    veto_list = f.readlines()

    veto_list = [v.replace('\n', '') for v in veto_list]

    files_processed = veto_list

if __name__ == "__main__":
    descr = 'process new all-sky camera images in real time'

    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument('data_dir', type=str,
                        help="directory with all-sky camera images")

    parser.add_argument('--outdir', default='.', type=str,
                        help="directory to write pipeline outputs in")

    parser.add_argument('--wait_seconds', default=5, type=int,
                        help="polling interval in seconds")

    parser.add_argument('--veto_list', default=None, type=str,
                        help="list of files not to process")

    args = parser.parse_args()

    assert(args.wait_seconds >= 1)

    if args.veto_list is not None:
        _do_veto(args.veto_list)

    _watch(args.data_dir, wait_seconds=args.wait_seconds,
           outdir=args.outdir)