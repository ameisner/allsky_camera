import numpy as np

def djs_photcen(xcen, ycen, image, cbox=7, cmaxiter=10, cmaxshift=0.0,
                ceps=0.0):
    # remember to output qmaxshift as well

    # for now assume xcen, ycen are each scalars

    # output needs to encapsulate three things :
    #     actual xcen, ycen recentered results
    #     qmaxshift boolean flag

    dims = image.shape
    assert(len(dims) == 2)
    naxis1 = dims[1] # numpy indexing convention; "x" direction
    naxis2 = dims[0] # numpy indexing convention; "y" direction

    Radius = 0.5*cbox

    iiter = 0
    dcen = 2*ceps + 1.0
    qmaxshift = 0

    ### my own code here
    xcen_orig = xcen
    ycen_orig = ycen
    ###
    
    while (iiter <= cmaxiter) and (qmaxshift == 0) and (np.max(np.abs(dcen)) > ceps):
        if (iiter > 0):
            dcen = np.array([xcen, ycen])

        xRad = min([Radius, max(xcen, 0), max(naxis1-xcen, 0)])
        iStart = np.floor(xcen + 0.5 - xRad)
        iEnd = np.ceil(xcen - 0.5 + xRad) + 1 # + 1 is for numpy indexing

        iEnd = min(iEnd, naxis1) # this isn't in IDL version, which is concerning

        if (iStart >= naxis1) or (iEnd <= 0):
            print('Error - No pixels in X range')
            return xcen_orig, ycen_orig, 1
        iLen = iEnd - iStart # note the lack of "+ 1" at end here !

        yRad = min([Radius, max(ycen, 0), max(naxis2-ycen, 0)])
        jStart = np.floor(ycen + 0.5 - yRad)
        jEnd = np.ceil(ycen - 0.5 + yRad) + 1 # + 1 is for numpy indexing

        jEnd = min(jEnd, naxis2) # this isn't in IDL version, which is concerning

        if (jStart >= naxis2) or (jEnd <= 0):
            print('Error - No pixels in Y range')
            return xcen_orig, ycen_orig, 1
        jLen = jEnd - jStart # note the lack of "+ 1" at end here !

        xA = iStart + np.arange(iLen) - 0.5 - xcen
        xB = xA + 1
        yA = jStart + np.arange(jLen) - 0.5 - ycen
        yB = yA + 1

        xA = np.maximum(xA, -1*xRad)
        xB = np.minimum(xB, xRad)
        yA = np.maximum(yA, -1*yRad)
        yB = np.minimum(yB , yRad)

        nPixelx = len(xA)
        nPixely = len(yA)
        nPixels = nPixelx*nPixely
        iIndx = np.arange(nPixels) % nPixelx
        jIndx = np.arange(nPixels) // nPixelx
        xPixNum = (iIndx + iStart).astype(int)
        yPixNum = (jIndx + jStart).astype(int)
        pixnum = xPixNum + naxis1*yPixNum
    
        fracs = (xB[iIndx] - xA[iIndx])*(yB[jIndx] - yA[jIndx])

        subimg = image[yPixNum, xPixNum]

        norm = np.sum(subimg*fracs)

        if norm > 0:
            xcen = np.sum(subimg*(xPixNum-xcen)*fracs)/norm + xcen
            ycen = np.sum(subimg*(yPixNum-ycen)*fracs)/norm + ycen

        if cmaxshift != 0:
            x_shift = xcen - xcen_orig
            y_shift = ycen - ycen_orig

            if (np.abs(x_shift) > cmaxshift) or (np.abs(y_shift) > cmaxshift):
                xcen = xcen_orig
                ycen = ycen_orig
                qmaxshift = 1

        if (iiter > 0):
            dcen = dcen - np.array([xcen, ycen])
        iiter += 1

    return xcen, ycen, qmaxshift # fill this in later
# want a wrapper function that takes list of (x, y) pairs and then
# does a for loop over the list of pairs, calling djs_photcen once
# for each pair

def _loop_djs_photcen(xcen, ycen, image, cbox=7, cmaxiter=10, cmaxshift=0.0,
                      ceps=0.0):

    from astropy.table import Table

    # assume cmaxiter, cmaxshift, ceps are scalar
    # cbox can be an array though
    assert(len(xcen) == len(ycen))

    n = len(xcen)

    if isinstance(cbox, float) or isinstance(cbox, int):
        cbox = [cbox]*n
    else:
        assert(len(cbox) == n)

    x_djs = np.zeros(n, dtype=float)
    y_djs = np.zeros(n, dtype=float)
    qmaxshift = np.zeros(n, dtype=int)
    for i in range(n):
        result = djs_photcen(xcen[i], ycen[i], image, cbox=cbox[i],
                             cmaxiter=cmaxiter, cmaxshift=cmaxshift,
                             ceps=ceps)

        x_djs[i] = result[0]
        y_djs[i] = result[1]
        qmaxshift[i] = result[2]

    results = Table()
    results['x_djs'] = x_djs
    results['y_djs'] = y_djs
    results['qmaxshift'] = qmaxshift
    results['x_start'] = xcen
    results['y_start'] = ycen
    results['cbox'] = cbox

    return results
