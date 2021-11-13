import multiprocessing
from scipy.ndimage import median_filter
import time
import numpy as np
from multiprocessing import Pool
import multiprocessing

def ind_split_image(im, nchunks, ksize=23, axis=1):
    """

    Parameters
    ----------
        im : numpy.ndarray
            2D image as a numpy array.
        nchunks : int
            Number of chunks into which to split the image along axis.
        ksize : int (optional)
            Median filter kernel size that will be applied downstream. In practice
            this dictates the amount of padding needed for each chunk, which is
            roughly ksize/2 pixels along each edge. ksize=0 is meant to behave
            such that the indices returned split the image into
            a set of mutually exclusive chunks.
        axis: int (optional)
            Axis along which it is desired to segment the image.
            Should be either 0 or 1, given that this function is intended
            for 2D images. Default is 1 (x axis).

    Returns
    -------
        starts : list
            List of starting indices of each image chunk. Length will be nchunks.
        ends : list
            List of ending indices of each image chunk (Python convention).
            Length will be nchunks.

    Notes
    -----
        Add a check to make sure nchunks is not larger than the image dimension along the
        axis of splitting? Other potential checks along these lines?
    """

    # nchunks=1 is a trivial case that shouldn't
    # actually be requested, but the correct answer is returned
    # in that case nevertheless
    assert(nchunks >= 1)

    pad = np.ceil(ksize/2)

    sh = im.shape

    assert(len(sh) == 2)

    npix = sh[axis]

    # + 1 is to be cautious/generous, may not be needed
    # the ksize != 0 part is there to make it so that ksize=0
    # results in mutually exclusive image chunks
    khalf = (ksize // 2) + 1*(ksize != 0)

    # starting indices of each chunk along the axis being segmented
    starts = []
    # ending indices of each chunk along the axis being segmented (Python convention)
    ends = []

    # size of each chunk in pixels
    size_approx = int(np.ceil(npix/float(nchunks)))

    for i in range(nchunks):
        start = max(i*size_approx - khalf, 0)
        end = min((i+1)*size_approx + khalf, npix)

        if i == (nchunks-1):
            end = npix

        starts.append(start)
        ends.append(end)

    return starts, ends

def split_and_reassemble(im, nchunks=2, ksize=23, nmp=None):
    """

    Parameters
    ----------
        im : numpy.ndarray
            2D image as a numpy array.
        nchunks : int (optional)
            Number of chunks into which to split the image along axis.
        ksize : int (optional)
            Median filter kernel size (sidelength of square kernel,
            in units of pixels) that will be applied downstream.
        nmp : int (optional)
            Number of threads for multiprocessing. Defaults to be
            the same as the number of image chunks requested. Should
            not be larger than the number of threads available on
            the machine being used.

    Returns
    -------
        stitched : numpy.ndarray
            2D image with same dimensions as input im, median filtered
            according to ksize kernel sidelength parameter.

    Notes
    -----
        Do I want to insist that ksize be an odd integer?
        Can only split into chunks segmented along the x axis for now.

    """


    # axis is unfortunately hardcoded for now
    # would be nice to generalize this to work for either
    # axis, but would require some changes to how the
    # indexing is treated below. Could also accomplish with
    # a transposition hack...
    axis = 1
    sh = im.shape
    assert(len(sh) == 2)

    if nmp is None:
        nmp = nchunks

    assert(nmp > 1)
    assert(nmp <= multiprocessing.cpu_count())

    starts_pad, ends_pad = ind_split_image(im, nchunks, ksize=ksize, axis=axis)

    # ksize=0 means splitting image into n chunks with
    # no padding/overlap of the chunks
    starts, ends = ind_split_image(im, nchunks, ksize=0, axis=1)

    stitched = np.zeros(im.shape)

    args = [(im[:, starts_pad[i]:ends_pad[i]], ksize) for i in range(nchunks)]

    print('Splitting image into ' + str(nmp) + ' chunks prior to median filter')
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