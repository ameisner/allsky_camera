import allsky_camera.util as util
import matplotlib.pyplot as plt
import astropy.io.fits as fits

im, h = fits.getdata('2020_10_11__21_38_23-detrended.fits', header=True)

mag_per_sq_asec = util.sky_brightness_map(im, h['EXPOSURE'])

plt.imshow(mag_per_sq_asec, origin='lower', cmap='gray_r', interpolation='nearest')

plt.show()
