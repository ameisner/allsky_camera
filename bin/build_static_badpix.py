"""
Script to build static bad pixel mask for MDM camera.
"""

import astropy.io.fits as fits
import glob
import os
import numpy as np
import allsky_camera.common as common
import fitsio
import scipy.signal as signal

dir = '/global/cfs/cdirs/desi/users/ameisner/MDM/allsky'

flist = glob.glob(os.path.join(dir, '*.fits'))

flist.sort()

par = common.ac_params()
cube = np.zeros((len(flist), par['ny'], par['nx']))

for i, f in enumerate(flist):
    im = fits.getdata(f)

    cube[i, :, :] = im

im_med = np.median(cube, 0)
medfilt = signal.medfilt(im_med, 3)

bad = (np.abs(im_med - medfilt) >= 10) | (im_med < 0)

bad = bad.astype(int)

outname = os.path.join(os.environ[par['meta_env_var']],
                       par['static_mask_filename'])

print(outname)

if os.path.exists(outname):
    print('file already exists, not overwriting...')
else:
    print('Writing : ' + outname)
    fitsio.write(outname, bad)
