import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage

def InverseCompositionAffine(It, It1):
    # Input:
    #	It: template image
    #	It1: Current image

    # Output:
    #	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    dx = scipy.ndimage.sobel(It, axis=1)
    dy = scipy.ndimage.sobel(It, axis=0)

    hh, ww = It.shape
    xrange_int = np.arange(0, ww)
    yrange_int = np.arange(0, hh)

    yy, xx = np.meshgrid(yrange_int, xrange_int)
    yy = yy.flatten()
    xx = xx.flatten()
    points = np.stack([xx, yy, np.ones_like(yy)], axis=1).copy()
    rbs_zt = RectBivariateSpline(yrange_int, xrange_int, It)
    rbs_zi = RectBivariateSpline(yrange_int, xrange_int, It1)
    rbs_dx = RectBivariateSpline(yrange_int, xrange_int, dx)
    rbs_dy = RectBivariateSpline(yrange_int, xrange_int, dy)


    dwdp = np.zeros((2, len(xx), 6))
    dwdp[0, :, 0] = xx
    dwdp[0, :, 1] = yy
    dwdp[0, :, 2] = 1
    dwdp[1, :, 3] = xx
    dwdp[1, :, 4] = yy
    dwdp[1, :, 5] = 1
    zdx = rbs_dx.ev(yy, xx)
    zdy = rbs_dy.ev(yy, xx)

    didw = np.stack([zdx, zdy], axis=1)

    Aall = np.einsum('cnk,nc->nk', dwdp, didw)


    dp = np.zeros((6,))
    tol = 1e-2
    while True:
        M[0, :] += dp[:3]
        M[1, :] += dp[3:]

        points_wrapped = M.dot(points.T).T
        idx_points_valid = (points_wrapped[:, 0] > 0) & (points_wrapped[:, 0] < ww - 1) \
                           & (points_wrapped[:, 1] > 0) & (points_wrapped[:, 1] < hh - 1)
        xx_wrapped = points_wrapped[idx_points_valid, 0]
        yy_wrapped = points_wrapped[idx_points_valid, 1]
        yy = points[idx_points_valid, 1]
        xx = points[idx_points_valid, 0]

        zi = rbs_zi.ev(yy_wrapped, xx_wrapped)
        zt = rbs_zt.ev(yy, xx)
        A = Aall[idx_points_valid, :]
        b = zt - zi

        dp = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
        print('M', M, 'dp', dp)
        if np.linalg.norm(dp) < tol:
            break
    return M