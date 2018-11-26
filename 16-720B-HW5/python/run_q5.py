import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x, np.ones((train_x.shape[0], 1)), batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here

initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, 1024, params, 'output')


losses = []
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        out = forward(h3, params, 'output', sigmoid)

        loss = ((out - xb) ** 2).sum()/len(train_x)
        total_loss += loss
        grad = 2 * (out - xb)
        grad = backwards(grad, params, 'output', sigmoid_deriv)
        grad = backwards(grad, params, 'hidden2', relu_deriv)
        grad = backwards(grad, params, 'hidden', relu_deriv)
        backwards(grad, params, 'layer1', relu_deriv)

        # apply gradient
        for n_layer in ['layer1', 'hidden', 'hidden2', 'output']:
            for n_param in ['W' + n_layer, 'b' + n_layer]:
                params['m_' + n_param] = 0.9 * params['m_' + n_param] - learning_rate * params['grad_' + n_param]
                params[n_param] += params['m_' + n_param]

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate - 1:
        learning_rate *= 0.9
    losses.append((total_loss))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(max_iters), losses, 'g', marker='x')
ax.set_xlabel('iter')
ax.set_ylabel('MSE')
plt.show()
# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt

xb = valid_x[[100,400,800,1500,2200],:]
h1 = forward(xb, params, 'layer1', relu)
h2 = forward(h1, params, 'hidden', relu)
h3 = forward(h2, params, 'hidden2', relu)
out = forward(h3, params, 'output', sigmoid)
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(xb[i].reshape(32, 32).T)
    plt.subplot(2, 5, i+6)
    plt.imshow(out[i].reshape(32, 32).T)
plt.show()

from skimage.measure import compare_psnr
# evaluate PSNR
# Q5.3.2
total_psnr = 0
for x in valid_x:
    h1 = forward(x, params, 'layer1', relu)
    h2 = forward(h1, params, 'hidden', relu)
    h3 = forward(h2, params, 'hidden2', relu)
    out = forward(h3, params, 'output', sigmoid)
    total_psnr += compare_psnr(x, out)
print('psnr', total_psnr/len(valid_x))