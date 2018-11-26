import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # apply threshold
    gray = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(gray)
    bw = skimage.morphology.closing(gray < thresh, skimage.morphology.square(11))

    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(bw)

    # label image regions
    label_image = skimage.measure.label(cleared)
    image_label_overlay = skimage.color.label2rgb(label_image, image=gray)

    # import matplotlib.pyplot as plt
    # import matplotlib.patches as mpatches
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            bboxes.append(region.bbox)
            # minr, minc, maxr, maxc = region.bbox
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                           fill=False, edgecolor='red', linewidth=2)
            # ax.add_patch(rect)

    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    return bboxes, bw
