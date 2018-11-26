import numpy as np
import scipy.io
from nn import *

import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

# from mpl_toolkits.axes_grid1 import ImageGrid
#
# fig = plt.figure()
# grid1 = ImageGrid(fig, 121, nrows_ncols=(8, 8), axes_pad=0.1, cbar_mode='single')
# grid2 = ImageGrid(fig, 122, nrows_ncols=(8, 8), axes_pad=0.1, cbar_mode='single')
# for i in range(64):
#     grid1[i].imshow(train_x[int(10800*np.random.random()), :].reshape((32, 32)).T, cmap='gray')
#     grid2[i].imshow(train_x[int(10800*np.random.random()), :].reshape((32, 32)).T, cmap='gray')
# plt.show()

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 5e-3
hidden_size = 64

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)
batches_test = get_random_batches(valid_x, valid_y, batch_size)
batch_num_test = len(batches_test)

params = {}

# initialize layers here

initialize_weights(1024, hidden_size, params, 'fc1')
initialize_weights(hidden_size, 36, params, 'fc2')

rb_before = params['Wfc1'].copy()

q312_data = []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss_train = 0
    avg_acc_train = 0
    total_loss_test = 0
    avg_acc_test = 0
    for xb, yb in batches:
        # forward
        rb = forward(xb, params, name='fc1')
        probs = forward(rb, params, name='fc2', activation=softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss_train += loss
        avg_acc_train += acc
        # backward
        # Implement backwards!
        dy = probs.copy()
        dy[np.arange(probs.shape[0]), yb.argmax(axis=1)] -= 1
        db = backwards(dy, params, 'fc2', linear_deriv)
        backwards(db, params, 'fc1', sigmoid_deriv)
        # apply gradient
        params['Wfc1'] = params['Wfc1'] - learning_rate * params['grad_Wfc1']
        params['Wfc2'] = params['Wfc2'] - learning_rate * params['grad_Wfc2']
        params['bfc1'] = params['bfc1'] - learning_rate * params['grad_bfc1']
        params['bfc2'] = params['bfc2'] - learning_rate * params['grad_bfc2']
        # print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, loss, acc))
    avg_acc_train /= batch_num
    total_loss_train /= float(len(train_x))
    for xb, yb in batches_test:
        # forward
        rb = forward(xb, params, name='fc1')
        probs = forward(rb, params, name='fc2', activation=softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss_test += loss/float(len(valid_x))
        avg_acc_test += acc / batch_num_test
    if itr % 2 == 0:
        print("itr: {:02d} \t train_loss: {:.2f} \t train_acc : {:.2f}\t test_loss {:.2f}\t test_acc {:.2f}".format(
            itr, total_loss_train, avg_acc_train, total_loss_test, avg_acc_test))
    q312_data.append((itr, total_loss_train, avg_acc_train, total_loss_test, avg_acc_test))

rb_after = params['Wfc1'].copy()
import pickle

saved_params = {k: v for k, v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.2
q312_data = np.array(q312_data)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
h1 = ax.plot(q312_data[:, 0], q312_data[:, 2], color='r', label='Train')
h2 = ax.plot(q312_data[:, 0], q312_data[:, 4], color='g', label='Test')
ax.legend(handles=h1 + h2)
ax.set_title('Accuracy change by epoch')
ax = fig.add_subplot(122)
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
h1 = ax.plot(q312_data[:, 0], q312_data[:, 1], color='r', label='Train')
h2 = ax.plot(q312_data[:, 0], q312_data[:, 3], color='g', label='Test')
ax.legend(handles=h1 + h2)
ax.set_title('Cross-entropy loss change by epoch')
plt.show()
# Q3.1.3
# from mpl_toolkits.axes_grid1 import ImageGrid
#
# fig = plt.figure()
# grid1 = ImageGrid(fig, 121, nrows_ncols=(8, 8), axes_pad=0.1, cbar_mode='single')
# grid2 = ImageGrid(fig, 122, nrows_ncols=(8, 8), axes_pad=0.1, cbar_mode='single')
# for i in range(hidden_size):
#     grid1[i].imshow(rb_before[:, i].reshape((32, 32)), cmap='gray')
#     grid2[i].imshow(rb_after[:, i].reshape((32, 32)), cmap='gray')
# plt.show()

# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1], train_y.shape[1]))
for xb, yb in batches:
    rb = forward(xb, params, name='fc1')
    probs = forward(rb, params, name='fc2', activation=softmax)
    yb_predict = probs.argmax(axis=1)
    yb = yb.argmax(axis=1)
    for y, yp in zip(yb, yb_predict):
        confusion_matrix[int(y), int(yp)] += 1

import string

plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
