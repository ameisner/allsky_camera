"""
allsky_camera.common
====================

Central location for parameters expected to be used in
various other places throughout the codebase.

Goal is to factor out special numbers.
"""

import yaml
from pkg_resources import resource_filename
import os

def ac_params(camera='MDM', verbose=False):
    """
    Provide a dictionary of various special numbers and other parameters

    Parameters
    ----------
        camera : str (optional)
            Name of specific all-sky camera. For now only 'MDM', the default,
            is supported.
        verbose : bool (optional)
            Whether or not to issue a printout about the YAML file name.


    Returns
    -------
        par : dict
            Dictionary of parameters.

    Notes:
        Need to think about the right way to avoid repeatedly reading
        in the YAML file within each end-to-end pipeline run.

    """

    fname_camera = resource_filename('allsky_camera',
                                     os.path.join('data', camera + '.yaml'))

    if verbose:
        print('READING ' + fname_camera)

    with open(fname_camera, 'r') as stream:
        par_camera = yaml.safe_load(stream)

    # miscellaneous parameters not tied to a particular camera
    par_misc = {'jd_minus_mjd' : 2400000.5,
                'bsc_filename' : 'bsc5.fits',
                'bsc_filename_csv' : 'bsc5.csv',
                'kpno_lon' : -111.5997,
                'kpno_lat' : 31.9599,
                'kpno_elev_meters' : 2100.0,
                'pressure_bars' : 1.013,
                'iso_thresh_deg': 0.3}

    par = dict(par_camera, **par_misc)
    return par
