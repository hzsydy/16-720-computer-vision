"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
from helper import *
from helper import _objective_F
import cv2
import sympy
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1_scaled = pts1.astype(np.float) / M
    pts2_scaled = pts2.astype(np.float) / M

    n = pts1.shape[0]
    A = np.zeros((n, 9))
    A[:, 2::3] = 1
    A[:, 1::3] = pts1_scaled[:, 1:2].copy()
    A[:, 0::3] = pts1_scaled[:, 0:1].copy()
    A[:, 0:3] *= pts2_scaled[:, 0:1]
    A[:, 3:6] *= pts2_scaled[:, 1:2]

    u, s, vh = np.linalg.svd(A.T.dot(A))
    f = u[:, -1].reshape((3, 3))

    f = refineF(f, pts1_scaled, pts2_scaled)
    t = np.diag([1. / M, 1. / M, 1])  # the scalar matrix
    return t.dot(f).dot(t)


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''


def sevenpoint(pts1, pts2, M):
    pts1_scaled = pts1.astype(np.float) / M
    pts2_scaled = pts2.astype(np.float) / M
    t = np.diag([1. / M, 1. / M, 1])  # the scalar matrix

    n = pts1.shape[0]
    A = np.zeros((7, 9))
    A[:, 2::3] = 1
    A[:, 1::3] = pts1_scaled[:, 1:2].copy()
    A[:, 0::3] = pts1_scaled[:, 0:1].copy()
    A[:, 0:3] *= pts2_scaled[:, 0:1]
    A[:, 3:6] *= pts2_scaled[:, 1:2]

    u, s, vh = np.linalg.svd(A.T.dot(A))
    f1 = u[:, -1].reshape((3, 3))
    f2 = u[:, -2].reshape((3, 3))

    # solve det(a*f1+(1-a)*f2)=0
    m1 = sympy.Matrix(f1)
    m2 = sympy.Matrix(f2)
    a = sympy.Symbol('a')
    m = a * m1 + (1 - a) * m2
    roots = sympy.solvers.solve(m.det(), a)

    result = []
    for root in roots:
        real, imagine = root.as_real_imag()
        a = float(real)
        if abs(float(imagine)) < 1e-3:
            f = a * f1 + (1 - a) * f2
            f = refineF(f, pts1_scaled, pts2_scaled)
            f = t.dot(f).dot(t)
            result.append(f)

    return result


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''


def essentialMatrix(F, K1, K2):
    return K2.T.dot(F).dot(K1)


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


def triangulate(C1, pts1, C2, pts2):
    assert (C1.shape == (3, 4))
    assert (C2.shape == (3, 4))
    assert (pts1.shape[1] == 2)
    assert (pts2.shape[1] == 2)
    n = pts1.shape[0]
    A = np.zeros((4, 4))
    P = np.zeros((n, 3))
    err = 0
    for i in range(n):
        A[0, :] = pts1[i, 0] * C1[2, :] - C1[0, :]
        A[1, :] = pts1[i, 1] * C1[2, :] - C1[1, :]
        A[2, :] = pts2[i, 0] * C2[2, :] - C2[0, :]
        A[3, :] = pts2[i, 1] * C2[2, :] - C2[1, :]

        u, s, vh = np.linalg.svd(A.T.dot(A))
        p = u[:, -1]
        p /= p[-1]
        P[i, :] = p[:-1].copy()

        reproj = C1.dot(p)
        reproj /= reproj[-1]
        err1 = reproj[:2] - pts1[i, :2]
        err += np.sum(err1 ** 2)
        reproj = C2.dot(p)
        reproj /= reproj[-1]
        err2 = reproj[:2] - pts2[i, :2]
        err += np.sum(err2 ** 2)

    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    M = max(*im1.shape)

    # get the epipolar line ax+by+c=0
    tilde1 = np.array([[x1, y1, 1]])
    a, b, c = tilde1.dot(F.T)[0]
    # a = F[0, 0] * x1 + F[1, 0] * y1 + F[2, 0]
    # b = F[0, 1] * x1 + F[1, 1] * y1 + F[2, 1]
    # c = F[0, 2] * x1 + F[1, 2] * y1 + F[2, 2]

    kx = a / np.sqrt(a * a + b * b)
    ky = b / np.sqrt(a * a + b * b)
    # get a initial point on ax+by+c=0 closest to (x1,y1)
    x20 = x1 - (a * x1 + b * y1 + c) / np.sqrt(a * a + b * b) * kx
    y20 = y1 - (a * x1 + b * y1 + c) / np.sqrt(a * a + b * b) * ky

    assert (np.abs(a * x20 + b * y20 + c) < 1e-4)

    # search
    window_size = 3
    import scipy.ndimage.filters as fi
    def gkern2(kernlen):
        inp = np.zeros((kernlen, kernlen))
        inp[kernlen // 2, kernlen // 2] = 1
        return fi.gaussian_filter(inp, sigma=3)

    weight = gkern2(window_size * 2 + 1)[:, :, None]
    window_template = im1[y1 - window_size:y1 + window_size + 1, x1 - window_size:x1 + window_size + 1, :]

    min_error = float('inf')
    min_p = (None, None)
    for delta in range(-20, 20):
        x2 = x20 + delta * ky
        y2 = y20 - delta * kx
        x2 = int(x2 + 0.5)
        y2 = int(y2 + 0.5)
        window = im2[y2 - window_size:y2 + window_size + 1, x2 - window_size:x2 + window_size + 1, :]
        print('window.shape', window.shape)
        error = np.abs(window_template.astype(np.float) - window.astype(np.float)) * weight
        error = error.sum()
        if error < min_error:
            min_error = error
            min_p = (x2, y2)

    return min_p
    # return x20, y20


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''


def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    min_F = None
    min_error = float('inf')
    for i in range(5000):
        F = sevenpoint(pts1, pts2, M)
        num_points = pts1.shape[0]

        error = _objective_F(F, pts1, pts2)
        if min_error>error:
            min_error = error
            min_F = F

    return min_F,


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    # Replace pass by your implementation
    pass


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    # Replace pass by your implementation
    pass


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass


def main():
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    N = pts1.shape[0]
    M = max(*im1.shape)

    ### test 2.1
    # f = eightpoint(pts1, pts2, M)
    # print(f)
    # np.savez('q2_1.npz', F=f, M=M)
    # displayEpipolarF(im1, im2, f)
    ### test 2.2
    # idx = np.random.randint(0, N, (7,))
    # fs = sevenpoint(pts1[idx,:], pts2[idx,:], M)
    # for f in fs:
    #     try:
    #         print(f)
    #         np.savez('q2_2.npz', F=f, M=M)
    #         displayEpipolarF(im1, im2, f)
    #     except Exception:
    #         pass
    ### test 4.1
    # F = eightpoint(pts1, pts2, M)
    # np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)
    # epipolarMatchGUI(im1, im2, F)

    ### test 4.2
    #data = np.load('../data/templeCoords.npz')
    #x1s = data['x1']
    #y1s = data['y1']
    #data = np.load('../data/intrinsics.npz')
    #K1 = data['K1']
    #K2 = data['K2']
    #data = np.load('q3 3.npz')
    #M2 = data['M2']
    #C2 = data['C2']
    #M1 = np.zeros((3, 4))
    #M1[[0, 1, 2], [0, 1, 2]] = 1.
    #C1 = K1.dot(M1)
    #F = eightpoint(pts1, pts2, M)
    #N = x1s.shape[0]
    #p1 = np.zeros((N, 2), dtype=np.int)
    #p2 = np.zeros((N, 2))
    #p1[:, 0:1] = x1s
    #p1[:, 1:2] = y1s
    #for i, p in enumerate(p1):
    #    x1, y1 = p
    #    x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
    #    p2[i] = x2, y2
#
    #np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)
#
    #P, err = triangulate(C1, p1, C2, p2)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1)
    #ax.set_aspect(1)
    #plt.show()

    ### test 5.1



if __name__ == '__main__':
    main()
