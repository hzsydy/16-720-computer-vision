import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    H = H2to1
    h, w = im1.shape[:2]
    im2_wrapped = cv2.warpPerspective(im2, H, (w, h))

    pano_im = im2_wrapped
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    '''
    H = H2to1

    h, w = im2.shape[:2]
    corner = np.array([
        [0, w - 1, 0, w - 1],
        [0, 0, h - 1, h - 1],
        [1, 1, 1, 1],
    ])
    corner_wrapped = H.dot(corner)
    corner_wrapped /= corner_wrapped[2:, :]
    corner_wrapped = (corner_wrapped[:2, :] + 0.5).astype(np.int)

    h, w = im1.shape[:2]
    x_min = min(corner_wrapped[0].min(), 0)
    x_max = max(corner_wrapped[0].max(), w)
    y_min = min(corner_wrapped[1].min(), 0)
    y_max = max(corner_wrapped[1].max(), h)

    new_x = x_max - x_min
    new_y = y_max - y_min
    dx = -x_min
    dy = -y_min
    assert (dy >= 0)
    assert (dx >= 0)
    new_H = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ]).dot(H)
    im2_wrapped = cv2.warpPerspective(im2, new_H, (new_x, new_y))
    #im2_wrapped[dy:dy + h, dx:dx + w, :] = im1
    pano_im = im2_wrapped

    # i1 = im1.astype(np.float)
    # i2 = im2_wrapped.astype(np.float)
    # i2[dy:dy + h, dx:dx + w, :][i2[dy:dy + h, dx:dx + w, :]>1e-3] /= 2
    # i2[dy:dy + h, dx:dx + w, :][i2[dy:dy + h, dx:dx + w, :]>1e-3] += i1[i2[dy:dy + h, dx:dx + w, :]>1e-3]/2
    # i2[dy:dy + h, dx:dx + w, :][i2[dy:dy + h, dx:dx + w, :]<1e-3] = i1[i2[dy:dy + h, dx:dx + w, :]<1e-3]
    # pano_im = i2.astype(np.uint8)
    return pano_im


def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # np.save('../results/q6_1.npy', H2to1)
    # pano_im = imageStitching(im1, im2, H2to1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
