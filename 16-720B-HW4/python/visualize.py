'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import *

im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
N = pts1.shape[0]
M = max(*im1.shape)
M1 = np.zeros((3, 4))
M1[[0, 1, 2], [0, 1, 2]] = 1.

data = np.load('../data/templeCoords.npz')
x1s = data['x1']
y1s = data['y1']
data = np.load('../data/intrinsics.npz')
K1 = data['K1']
K2 = data['K2']
data = np.load('q3_3.npz')
M2 = data['M2']
C2 = data['C2']
C1 = K1.dot(M1)
F = eightpoint(pts1, pts2, M)
N = x1s.shape[0]
p1 = np.zeros((N, 2), dtype=np.int)
p2 = np.zeros((N, 2))
p1[:, 0:1] = x1s
p1[:, 1:2] = y1s
for i, p in enumerate(p1):
   x1, y1 = p
   x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
   p2[i] = x2, y2


#np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

P, err = triangulate(C1, p1, C2, p2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1)
ax.set_aspect(1)
plt.show()