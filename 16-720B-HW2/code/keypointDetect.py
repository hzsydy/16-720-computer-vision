import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1,
                          k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im) / 255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0 * k ** i
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''

    DoG_levels = levels[1:]
    h, w, _ = gaussian_pyramid.shape
    DoG_pyramid = np.zeros((h, w, len(DoG_levels)))
    for i, l in enumerate(DoG_levels):
        DoG_pyramid[:, :, i] = gaussian_pyramid[:, :, l + 1] - gaussian_pyramid[:, :, l]

    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    h, w, c = DoG_pyramid.shape
    principal_curvature = np.zeros((h, w, c))
    for idx in range(c):
        im = DoG_pyramid[:, :, idx]
        im_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        im_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        im_xx = cv2.Sobel(im_x, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        im_xy = cv2.Sobel(im_x, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        im_yx = cv2.Sobel(im_y, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        im_yy = cv2.Sobel(im_y, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        r = (im_xx + im_yy) ** 2 / (im_xx * im_yy - im_xy * im_yx)

        # assert (np.allclose(im_xy[1:-1], im_yx[1:-1], rtol=1e-3, atol=1e-3))

        principal_curvature[:, :, idx] = r

    return np.abs(principal_curvature)   # https://piazza.com/class/jl5eyuqvtrc3d0?cid=277


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
                    th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''

    # the 10 neighbouts
    #a = DoG_pyramid
    #l = np.roll(a, -1, axis=1)
    #l[:, -1, :] = -np.inf
    #r = np.roll(a, 1, axis=1)
    #r[:, 0, :] = -np.inf
    #t = np.roll(a, -1, axis=0)
    #t[-1, :, :] = -np.inf
    #b = np.roll(a, 1, axis=0)
    #b[0, :, :] = -np.inf
    #l#t = np.roll(l, -1, axis=0)
    #lt[-1, :, :] = -np.inf
    #rt = np.roll(r, -1, axis=0)
    #rt[-1, :, :] = -np.inf
    #lb = np.roll(l, 1, axis=0)
    #lb[0, :, :] = -np.inf
    #rb = np.roll(r, 1, axis=0)
    #rb[0, :, :] = -np.inf
    #u = np.roll(a, -1, axis=2)
    #u[:, :, -1] = -np.inf
    #d = np.roll(a, 1, axis=2)
    #d[:, :, 0] = -np.inf
    #res = np.ones_like(DoG_pyramid).astype(np.bool)
    #for n in [l, r, t, b, lt, rt, lb, rb, u, d]:
    #    res = np.logical_and(res, a > n)
    import scipy.ndimage
    footprint = np.zeros((3,3,3))
    footprint[:,:,1]=1
    footprint[1,1,:]=1
    res1 = scipy.ndimage.maximum_filter(DoG_pyramid, footprint=footprint)
    res2 = scipy.ndimage.minimum_filter(DoG_pyramid, footprint=footprint)
    res = np.logical_or(DoG_pyramid==res1, DoG_pyramid==res2)
    res = np.logical_and(res, np.abs(DoG_pyramid) > th_contrast)
    res = np.logical_and(res, principal_curvature < th_r)
    return np.array(np.where(res))[[1,0,2],:].T


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1, 0, 1, 2, 3, 4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector

    #im = cv2.imread('../data/chickenbroth_01.jpg')
    locsDoG, gaussian_pyramid = DoGdetector(im)

    h, w, c = gaussian_pyramid.shape
    im_pyramid = np.split(gaussian_pyramid, gaussian_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im_pyr_show = np.repeat(im_pyramid[:, :, None], 3, axis=2)
    im_show = np.repeat(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, :, None] / 255., 3, axis=2).astype(np.float32)
    for x, y, c in locsDoG:
        cv2.circle(im_pyr_show, (x + c * w, y), 1, (0, 1, 0), thickness=-1)
        cv2.circle(im_show, (x, y), 1, (0, 1, 0), thickness=-1)
    cv2.imshow('feature points in pyramid', im_pyr_show)

    cv2.imshow('feature points on image', cv2.resize(im_show, (2 * w, 2 * h)))
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()
