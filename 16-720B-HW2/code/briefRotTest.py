from BRIEF import *
import math

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    h, w = im1.shape[:2]
    center = (int(0.5 * w), int(0.5 * h))
    size = int(1.1 * math.sqrt(w * w + h * h))

    n = []

    for i in range(36):
        rotation = cv2.getRotationMatrix2D(center, 10 * i, 1)
        rotation[0, 2] += size // 2 - center[0]
        rotation[1, 2] += size // 2 - center[1]
        im2 = cv2.warpAffine(im1, rotation, (size, size))
        locs1, desc1 = briefLite(im1)
        locs2, desc2 = briefLite(im2)
        # print(desc1.shape, desc2.shape)
        matches = briefMatch(desc1, desc2)

        m1 = locs1[matches[:, 0]]
        m2 = locs2[matches[:, 1], :2]
        m1[:, 2] = 1
        m12 = rotation.dot(m1.T).T

        n.append(np.sum(np.linalg.norm(m2 - m12, axis=1) < 1.5))
        # draw two images side by side
        plt.subplot(6, 6, i + 1)
        plt.axis("off")
        imH = max(im1.shape[0], im2.shape[0])
        im = np.zeros((imH, im1.shape[1] + im2.shape[1]), dtype='uint8')
        im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        plt.imshow(im, cmap='gray')
        for i in range(matches.shape[0]):
            pt1 = locs1[matches[i, 0], 0:2]
            pt2 = locs2[matches[i, 1], 0:2].copy()
            pt2[0] += im1.shape[1]
            x = np.asarray([pt1[0], pt2[0]])
            y = np.asarray([pt1[1], pt2[1]])
            plt.plot(x, y, 'r')
            plt.plot(x, y, 'g.')
    plt.show()

    plt.bar([i * 10 for i in range(36)], n, width=2)
    plt.show()
