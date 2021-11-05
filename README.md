# allsky_camera

A pipeline for processing all-sky camera images.

This all-sky camera reduction/analysis pipeline is implemented in pure Python. It has a relatively small number of dependencies that must be installed.

The ALLSKY_CAMERA_META environment variable should point to a directory with ancillary calibration products.

Here is an example invocation for running the full pipeline:

    python -u /global/homes/a/ameisner/allsky_camera/py/allsky_camera/ac_proc.py /global/cfs/cdirs/desi/users/ameisner/MDM/allsky/2020_10_11__21_38_23.fits --outdir .

![zeropoint checkplot](static/2020_10_11__21_38_23-zp.png)
