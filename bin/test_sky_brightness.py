import allsky_camera.util as util
import allsky_camera.io as io
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from allsky_camera.exposure import AC_exposure

exp = AC_exposure('2020_10_11__21_38_23-detrended.fits')

util.detrend_ac(exp)

mag_per_sq_asec = util.sky_brightness_map(exp.detrended, exp.time_seconds)

io.sky_brightness_plot(mag_per_sq_asec, None, None)

#plt.show()
