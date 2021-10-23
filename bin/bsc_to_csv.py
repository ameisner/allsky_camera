"""
Script to convert BSC5 bright star catalog to CSV format.
"""

import astropy.io.fits as fits
from astropy.table import Table
import allsky_camera.common as common
import os

par = common.ac_params()

fname = os.path.join(os.environ[par['meta_env_var']],
                     par['bsc_filename'])

tab = fits.getdata(fname)

tab = Table(tab)

outname = fname.replace('.fits', '.csv')

assert(not os.path.exists(outname))

tab.write(outname, format='csv')
