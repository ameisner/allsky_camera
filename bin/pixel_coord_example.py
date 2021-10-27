"""

Minimal example illustrating how to instantiate a bright star catalog object
and convert the star (ra, dec) values to all-sky camera pixel coordinates
at a given epoch.

"""


import allsky_camera.starcat as starcat
import allsky_camera.util as util
import matplotlib.pyplot as plt

sc = starcat.StarCat()

mjd0 = 55229.0
mjd1 = mjd0 + 0.1

cat = sc.cat_with_pixel_coords(mjd0)
cat2 = sc.cat_with_pixel_coords(mjd1)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(cat['x'], cat['y'], s=2, edgecolor='none')

plt.subplot(1, 2, 2)
plt.scatter(cat2['x'], cat2['y'], s=2, edgecolor='none')

plt.show()
plt.cla()
