import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
import scipy.ndimage
import scipy
import cv2


def SubtractDominantMotion(image1, image2):
    # Input:
    #	Images at time t and t+1
    # Output:
    #	mask: [nxm]
    # put your implementation here
    M2to1 = InverseCompositionAffine(image2, image1)
    # M2to1 = LucasKanadeAffine(image2, image1)
    im2_wrapped = cv2.warpAffine(image2, M2to1, image2.shape[::-1])

    mask = np.isclose(im2_wrapped, image1, atol=0.1)
    mask = 1 - mask

    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=5)
    mask = scipy.ndimage.morphology.binary_erosion(mask, iterations=5)
    # mask = scipy.ndimage.morphology.binary_erosion(mask,iterations=3)
    # mask = scipy.ndimage.morphology.binary_dilation(mask,iterations=3)

    return mask
