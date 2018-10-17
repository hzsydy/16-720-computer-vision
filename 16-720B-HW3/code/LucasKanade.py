import numpy as np
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline
import math

def LucasKanade(It, It1, rect, p0=np.zeros(2)):
    # Input:
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here

    dx = scipy.ndimage.sobel(It1, axis=1)
    dy = scipy.ndimage.sobel(It1, axis=0)

    l, t, r, b = rect
    h, w = b - t, r - l
    hh, ww = It1.shape
    xrange_int = np.arange(0, ww)
    yrange_int = np.arange(0, hh)

    rbs_zt = RectBivariateSpline(yrange_int, xrange_int, It)
    xrange = np.arange(l, l + w - 1e-5, 1)
    yrange = np.arange(t, t + h - 1e-5, 1)
    yy, xx = np.meshgrid(yrange, xrange)
    yy = yy.flatten()
    xx = xx.flatten()
    zt = rbs_zt.ev(yy, xx)
    rbs_zi = RectBivariateSpline(yrange_int, xrange_int, It1)
    rbs_dx = RectBivariateSpline(yrange_int, xrange_int, dx)
    rbs_dy = RectBivariateSpline(yrange_int, xrange_int, dy)

    p = p0
    dp = np.zeros((0, 0))
    tol = 0.1
    while True:
        xrange = np.arange(l + p[0], l + p[0] + w - 1e-5, 1)
        yrange = np.arange(t + p[1], t + p[1] + h - 1e-5, 1)
        yy, xx = np.meshgrid(yrange, xrange)
        yy = yy.flatten()
        xx = xx.flatten()

        zi = rbs_zi.ev(yy, xx)
        zdx = rbs_dx.ev(yy, xx)
        zdy = rbs_dy.ev(yy, xx)

        A = np.stack([zdx, zdy], axis=1)
        b = zt - zi

        dp = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
        p += dp
        print ('p', p, 'dp', dp)
        if np.linalg.norm(dp)<tol:
            break

    return p
