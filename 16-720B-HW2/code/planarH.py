import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
            equation
    '''
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    _, n = p1.shape
    x_ = np.concatenate([p1, np.ones((1, n))], axis=0).T
    u_ = np.concatenate([p2, np.ones((1, n))], axis=0).T

    A = np.zeros((2 * n, 9))
    A[0::2, 3:6] = -u_
    A[1::2, 0:3] = -u_
    A[1::2, 6:9] = u_ * x_[:, 0:1]
    A[0::2, 6:9] = u_ * x_[:, 1:2]

    M = A.T.dot(A)
    u, s, vh = np.linalg.svd(M)
    #i = np.argmin(s[s > 1e-3])
    H2to1 = u[:, -1].reshape((3, 3))

    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    n = matches.shape[0]
    idx = list(range(n))
    x_sb = locs1[matches[:, 0],:2].T
    u_sb = locs2[matches[:, 1],:2].T
    x = np.concatenate([x_sb, np.ones((1, n))], axis=0)
    u = np.concatenate([u_sb, np.ones((1, n))], axis=0)
    n_bestH = 0
    bestH = None
    for i in range(num_iter):
        np.random.shuffle(idx)
        idx_selected = idx[:4]

        H = computeH(x_sb[:, idx_selected], u_sb[:, idx_selected])
        error = H.dot(u)
        error = error/error[2:,:] - x
        error = error[:2,:]
        error = np.sqrt(np.sum(error*error, axis=0))
        if np.sum(error < tol)>n_bestH:
            n_bestH = np.sum(error < tol)
            bestH = H
        print ('n_bestH', n_bestH)
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
