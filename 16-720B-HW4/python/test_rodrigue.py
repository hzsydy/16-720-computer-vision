from submission import rodrigues as myRodrigues
from submission import invRodrigues as myInvRodrigues
from submission import rodriguesResidual as myrodriguesResidual
import submission as sub
import numpy as np
import unittest

def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    if theta>0:
        k = r/theta
        K = np.array([[0,-k[2],k[1]],\
            [k[2],0,-k[0]],\
            [-k[1],k[0],0]])
        R = np.eye(3) + np.sin(theta)*K + (1.-np.cos(theta))*np.dot(K,K)
    else:
        K = np.array([[0,-r[2],r[1]],\
            [r[2],0,-r[0]],\
            [-r[1],r[0],0]])
        theta2 = theta * theta
        R = np.eye(3) + (1. - theta*theta/6.)*K + (0.5 - theta*theta / 24.) * np.dot(K,K)

    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    t = np.trace(R)
    r = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    a = np.argmax(np.array([R[0,0],R[1,1],R[2,2]]))
    b = (a+1)%3
    c = (a+2)%3
    s = np.sqrt(R[a,a]-R[b,b]-R[c,c]+1)
    v = np.array([s/2., (R[a,b]+R[b,a])/(2.*s), (R[a,c]+R[c,a])/(2.*s)])
    theta = np.arccos((t-1)/2.)
    if t>=3-1e-30:
        w = r*(.5-(t-3)/12.)
    elif -1+1e-30<t<3-1e-30:
        w = r*(theta/(2*np.sin(theta)))
    else:
        w = np.pi*v/np.linalg.norm(v)
    return w




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
    N = p1.shape[0]
    C1 = K1.dot(M1)
    P = x[:3*N].reshape(N, 3)
    P_hat = np.hstack([P, np.ones((N, 1))])
    r2 = x[3*N: 3*N+3].reshape(-1,1)
    t2 = x[3*N+3:].reshape(-1,1)
    R2 = rodrigues(r2)
    # print('cv: ',cv2.Rodrigues(r2),'compute: ',R2)
    # print('RD: ',R2-cv2.Rodrigues(r2)[0])
    # print(N,x.shape)
    C2 = np.dot(K2, np.hstack([R2, t2]))

    p1_hat = C1.dot(P_hat.T)#3*N
    p1_hat /= p1_hat[-1,:]
    p1_hat = p1_hat[:-1,:].T
    p2_hat = C2.dot(P_hat.T) #3*N
    p2_hat /= p2_hat[-1,:]
    p2_hat = p2_hat[:-1,:].T
    # print(p1.shape,p1_hat.shape)
    # print(p2.shape,p2_hat.shape)

    residuals = np.expand_dims(np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])]),axis=0).T
    # residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals




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
    N = P_init.shape[0]
    r = invRodrigues(M2_init[:, :3]).reshape(-1)
    # print('cv: ',cv2.Rodrigues(M2_init[:, :3]),'compute: ',r)
    t = M2_init[:, 3].reshape(-1)
    x = np.concatenate((P_init.reshape(-1), r, t))
    # print(x.shape,r.shape,t.shape,P_init.shape)
    rodriguesResidualOptim = lambda x, K1, M1, p1, K2, p2: rodriguesResidual(K1, M1, p1, K2, p2, x).reshape(-1)
    import scipy
    res_x = scipy.optimize.leastsq(rodriguesResidualOptim, x, (K1, M1, p1, K2, p2))[0]
    # print(np.sum(res_x-x))
    P = res_x[:3*N].reshape(N, 3)
    r = res_x[3*N: 3*N+3].reshape(-1,1)
    t = res_x[3*N+3:].reshape(-1,1)
    R = rodrigues(r)
    M2 = np.hstack([R, t])
    return M2,P



class testRodrigue(unittest.TestCase):
    def test_rodrigue(self):
        r = np.random.randn(3,1)/2
        R = rodrigues(r)
        myR = myRodrigues(r)
        myR1 = myRodrigues(r.flatten())
        print(R)
        print(myR)
        print(R.T.dot(R))
        print(np.identity(3))
        self.assertTrue(np.allclose(R.T.dot(R), np.identity(3), rtol=1e-3))
        self.assertTrue(np.allclose(R, myR, rtol=1e-3))
        self.assertTrue(np.allclose(myR1, myR, rtol=1e-3))

    def test_inv_rodrigue(self):
        r = np.random.randn(3)/2
        R = myRodrigues(r)
        self.assertTrue(np.allclose(invRodrigues(R).flatten(), myInvRodrigues(R).flatten(), rtol=1e-3))

    def test_rodriguesResidual(self):
        K1 = np.random.rand(3, 3)
        K2 = np.random.rand(3, 3)
        M1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
        M2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
        r2 = np.ones(3)
        t2 = np.ones(3)
        P = np.random.randn(140, 3)
        x = np.concatenate([P.reshape([-1]), r2, t2])
        pts1 = np.random.randn(140, 2)
        pts2 = np.random.randn(140, 2)
        my_residuals = myrodriguesResidual(K1, M1, pts1, K2, pts2, x)
        residuals = rodriguesResidual(K1, M1, pts1, K2, pts2, x)
        self.assertTrue(np.allclose(residuals.flatten(), my_residuals.flatten(), rtol=1e-3))
