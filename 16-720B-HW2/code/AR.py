import numpy as np
import matplotlib.pyplot as plt
from planarH import computeH
import cv2


def compute_extrinsics(K, H):
    '''
    where K contains the intrinsic parameters and H contains the estimated homography
    :param K:intrinsic parameters
    :param H:estimated homography
    :return:
    '''

    h = np.linalg.inv(K).dot(H)
    u, _, vh = np.linalg.svd(h[:, :2])
    R = np.zeros((3, 3))
    R[:, :2] = u.dot(np.array([[1, 0], [0, 1], [0, 0]])).dot(vh)

    w3 = np.cross(R[:, 0], R[:, 1])
    w3 /= np.linalg.norm(w3)
    R[:, 2] = w3
    if np.linalg.det(R) == -1:
        R[:, 2] *= -1
    l = (h[:, :2] / R[:, :2]).mean()
    t = h[:, 2] / l

    return R, t


def project_extrinsics(K, W, R, t):
    '''
    :param K:
    :param W:
    :param R:
    :param t:
    :return: project the set of 3D points in the file sphere.txt(which has been fixed to have the
same radius as a tennis ball - of 6.858l)onto the image prince book.jpg
    '''

    X = R.dot(W) + t[:, None]
    X = K.dot(X)
    X /= X[2:, :]
    return X[:2, :]


if __name__ == '__main__':
    W = np.array([
        [0, 18.2, 18.2, 0],
        [0, 0, 26, 26],
        [0, 0, 0, 0]
    ])

    X = np.array([
        [483, 1704, 2175, 67],
        [810, 781, 2217, 2286]
    ])

    K = np.diag([3043.72, 3043.72, 1])
    K[0, 2] = 1196
    K[1, 2] = 1604

    H = computeH(X, W[:2, :])
    print(H)
    R, t = compute_extrinsics(K, H)
    print(R, t)

    w = np.loadtxt('../data/sphere.txt')
    w[0] += 5.5
    w[1] += 15.0
    x = project_extrinsics(K, w, R, t)
    x = x.astype(np.int)

    im = cv2.imread('../data/prince_book.jpeg')

    plt.imshow(im[:, :, ::-1])
    plt.scatter(x[0], x[1], color='yellow', s=1)
    plt.show()
