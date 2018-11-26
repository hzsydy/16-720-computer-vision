import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
u, s, vh = np.linalg.svd(train_x)
priciples = vh.T[:,:32]


# build valid dataset

coeff = valid_x.dot(priciples)
recon_valid = coeff.dot(priciples.T)

idx = [100,400,800,1500,2200]
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(valid_x[idx[i]].reshape(32, 32).T)
    plt.subplot(2, 5, i+6)
    plt.imshow(recon_valid[idx[i]].reshape(32, 32).T)
plt.show()

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())