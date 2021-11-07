# allsky_camera

A pipeline for processing all-sky camera images.

This all-sky camera reduction/analysis pipeline is implemented in pure Python. It has a relatively small number of dependencies that must be installed.

The ALLSKY_CAMERA_META environment variable should point to a directory with ancillary calibration products.

Here is an example invocation for running the full pipeline:

    python -u /global/homes/a/ameisner/allsky_camera/py/allsky_camera/ac_proc.py /global/cfs/cdirs/desi/users/ameisner/MDM/allsky/2020_10_11__21_38_23.fits --outdir .

This produces, among other output, the following photometry/zeropoint check plot:

![zeropoint checkplot](static/2020_10_11__21_38_23-zp.png)

The following checkplot of astrometric residuals relative to the static template is also produced:

![astrometry checkplot](static/2020_10_11__21_38_23-quiver.png)

The list of outputs produced is:

    2020_10_11__21_38_23-cat.fits
    2020_10_11__21_38_23-detrended.fits
    2020_10_11__21_38_23-zp.png

* The -cat.fits output is a FITS binary table with the source catalog including measured centroid locations and fluxes.
* The -detrended.fits output is a detrended version of the raw all-sky camera image.
* The -zp.png output is a checkplot showing the all-sky camera photometry and corresponding zeropoint measurement.

# full help for running the pipeline

    allsky_camera/py/allsky_camera> python ac_proc.py --help
    usage: ac_proc.py [-h] [--outdir OUTDIR] [--dont_write_detrended] [--multiproc MULTIPROC] fname_in

    run the all-sky camera reduction pipeline on an exposure

    positional arguments:
      fname_in              all-sky camera raw image file name

    optional arguments:
      -h, --help            show this help message and exit
      --outdir OUTDIR       directory to write outputs in
      --dont_write_detrended
                            don't write detrended image
      --multiproc MULTIPROC
                            number of threads for multiprocessing
