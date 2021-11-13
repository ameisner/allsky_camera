import multiprocessing
from scipy.ndimage import median_filter
import time
import numpy as np
from multiprocessing import Pool
import multiprocessing

def ind_split_image(im, nchunks, ksize=23, axis=1):

    pad = np.ceil(ksize/2)

    sh = im.shape

    assert(len(sh) == 2)

    npix = sh[axis]

    # + 1 is to be cautious/generous
    half = (ksize // 2) + 1*(ksize != 0)

    starts = []
    ends = []

    size_approx = int(np.ceil(npix/float(nchunks)))

    for i in range(nchunks):
        start = max(i*size_approx - half, 0)
        end = min((i+1)*size_approx + half, npix)

        if i == (nchunks-1):
            end = npix

        starts.append(start)
        ends.append(end)

    return starts, ends

def split_and_reassemble(im, nchunks=2, ksize=23, nmp=2):

    sh = im.shape
    assert(len(sh) == 2)

    assert(nmp > 1)
    assert(nmp <= multiprocessing.cpu_count())

    starts_pad, ends_pad = ind_split_image(im, nchunks, ksize=ksize, axis=1)

    # ksize=0 means splitting image into n chunks with
    # no padding/overlap of the chunks
    starts, ends = ind_split_image(im, nchunks, ksize=0, axis=1)

    stitched = np.zeros(im.shape)

    args = [(im[:, starts_pad[i]:ends_pad[i]], ksize) for i in range(nchunks)]

    p = Pool(nmp)

    t0 = time.time()
    print('starting parallelized median filter')
    med_chunks = p.starmap(median_filter, args)
    print('ending parallelized median filter...took ', time.time()-t0, ' seconds')

    p.close()

    for i in range(nchunks):
        chunk = med_chunks[i]
        stitched[:, starts[i]:ends[i]] = \
            chunk[:, (starts[i] - starts_pad[i]):(chunk.shape[1]-(ends_pad[i] - ends[i]))]

    return stitched

def _test():
    import astropy.io.fits as fits
    import allsky_camera.common as common

    im = fits.getdata('/Users/ameisner/MDM/allsky/2020_10_11__21_38_23.fits')

    par = common.ac_params()

    t0 = time.time()
    med_serial = median_filter(im, par['ksize'])
    print('vanilla median filter took ', time.time()-t0, ' seconds')

    stitched = split_and_reassemble(im, nchunks=2, ksize=23, nmp=2)

    assert(np.all(stitched == med_serial))