import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images', img)))
    bboxes, bw = findLetters(im1)


    # plt.imshow(im1)
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                         fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.

    def roi(bbox1, bbox2):
        ymin1, xmin1, ymax1, xmax1 = bbox1
        ymin2, xmin2, ymax2, xmax2 = bbox2
        if ymin2 > ymax1 or ymin1 > ymax2:
            return 0.
        return abs(max(ymin1, ymin2) - min(ymax1, ymax2)) / (max(ymax1, ymax2) - min(ymin1, ymin2))


    merged_bbox = []
    for bbox in bboxes:
        merge = False
        for bbox_group in merged_bbox:
            if roi(bbox, bbox_group[0]) > 0.2:
                bbox_group.append(bbox)
                bbox_group[0] = (min(bbox_group[0][0], bbox[0]), None, max(bbox_group[0][2], bbox[2]), None)
                merge = True
                break
        if not merge:
            merged_bbox.append([bbox, bbox])
    #
    # from pprint import pprint
    # pprint(merged_bbox)
    merged_bbox = [bbox_group[1:] for bbox_group in merged_bbox]

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))

    plt.imshow(im1)
    colors = 'bgrycmkwbgrcmykw'
    import sys
    for i, bbox_group in enumerate(merged_bbox):
        bbg = sorted(bbox_group, key=lambda x:x[1])
        for j, bbox in enumerate(bbg):
            ymin, xmin, ymax, xmax = bbox

            roi = 1 - bw[ymin:ymax + 1, xmin:xmax + 1].astype(np.float)
            size = 6 * max(ymax - ymin, xmax - xmin) // 5
            dy = (size - ymax + ymin) // 2
            dx = (size - xmax + xmin) // 2
            roi = np.pad(roi, ((dy, dy), (dx, dx)), 'constant', constant_values=1)
            roi = skimage.morphology.erosion(roi)
            roi = skimage.morphology.erosion(roi)
            roi = skimage.transform.resize(roi, (32, 32))


            x = roi.T.reshape((1, 1024))
            r = forward(x, params, name='fc1')
            probs = forward(r, params, name='fc2', activation=softmax)
            y = probs.argmax(axis=1)
            c = letters[y[0]]

            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                fill=False, edgecolor=colors[i], linewidth=2)
            plt.gca().add_patch(rect)
            rect = matplotlib.patches.Rectangle((xmin - dx, ymin - dy), xmax - xmin + 2 * dx,
                                                ymax - ymin + 2 * dy, fill=False, edgecolor='w', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(xmin, ymin - 20, c)
            if j>=1:
                ymin2, xmin2, ymax2, xmax2 = bbg[j-1]
                if xmin-xmax2>(xmax-xmin)*1.5:
                    sys.stdout.write(' ')
            sys.stdout.write(c)
        sys.stdout.write('\n')
        sys.stdout.flush()
    plt.show()
