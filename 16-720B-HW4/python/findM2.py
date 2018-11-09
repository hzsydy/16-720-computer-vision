'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

from submission import *

im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')

data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
N = pts1.shape[0]
M = max(*im1.shape)

data = np.load('../data/intrinsics.npz')
K1 = data['K1']
K2 = data['K2']

F = eightpoint(pts1, pts2, M)
E = essentialMatrix(F, K1, K2)

M1 = np.zeros((3, 4))
M1[[0, 1, 2], [0, 1, 2]] = 1.
M2s = camera2(E)

for i in range(4):
    M2 = M2s[:, :, i]
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P, err = triangulate(C1, pts1, C2, pts2)

    # check if P is legal
    if not np.alltrue(P[:, -1] > 0):
        continue
    else:
        print('==============')
        print(M2)
        print('err:', err)
        np.savez('q3 3.npz', M2=M2, C2=C2, P=P)
