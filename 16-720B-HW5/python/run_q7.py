import torch
import torchvision
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm

batch_size = 32
device = torch.device('cuda:0')
nr_train = 10800
nr_valid = 3600


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Net71(nn.Module):
    def __init__(self):
        super(Net71, self).__init__()
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 36)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class Net72(nn.Module):
    def __init__(self, nr_class):
        super(Net72, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.conv = nn.Conv2d(64, 256, kernel_size=4)
        self.fc = nn.Linear(256, nr_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv(x).view(x.shape[0], -1))
        return F.log_softmax(self.fc(x), dim=1)


def get_dataset_nist():
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

    train_x, train_y = train_data['train_data'], train_data['train_labels']
    valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
    train_tensor = torch.utils.data.TensorDataset(torch.tensor(train_x).float(),
                                                  torch.tensor(np.argmax(train_y, axis=1)).long())
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=1)

    valid_tensor = torch.utils.data.TensorDataset(torch.tensor(valid_x).float(),
                                                  torch.tensor(np.argmax(valid_y, axis=1)).long())
    valid_loader = DataLoader(valid_tensor, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader, valid_loader


def train(net, opt, loader):
    net.train()
    for batch_idx, (x, y_gt) in enumerate(loader):
        x = x.to(device)
        y_gt = y_gt.to(device)
        opt.zero_grad()
        prob = net(x)
        loss = F.nll_loss(prob, y_gt)
        loss.backward()
        opt.step()


def test(net, loader, average_factor):
    net.eval()
    acc = 0.0
    tot_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y_gt) in enumerate(loader):
            x = x.to(device)
            y_gt = y_gt.to(device)
            prob = net(x)
            loss = F.nll_loss(prob, y_gt, size_average=False)
            tot_loss += loss.item()
            _, y_pd = prob.max(1)
            acc += y_pd.eq(y_gt).sum().item()
    acc /= float(average_factor)
    tot_loss /= float(average_factor)
    return tot_loss, acc


def get_dataset_mnist():
    from torchvision import transforms

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    data = torchvision.datasets.MNIST(r'../MNIST', download=True, transform=transform)
    num_train = len(data)
    indices = list(range(num_train))

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=nr_valid, replace=False)
    indices = list(set(indices) - set(validation_idx))
    train_idx = np.random.choice(indices, size=nr_train, replace=False)

    train_loader = DataLoader(data, batch_size=batch_size, num_workers=1,
                              sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(data, batch_size=batch_size, num_workers=1,
                              sampler=SubsetRandomSampler(validation_idx))
    return train_loader, valid_loader


def get_dataset_nist_conv():
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

    train_x, train_y = train_data['train_data'], train_data['train_labels']
    valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
    train_tensor = torch.utils.data.TensorDataset(torch.tensor(train_x.reshape((-1, 1, 32, 32))).float(),
                                                  torch.tensor(np.argmax(train_y, axis=1)).long())
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=1)

    valid_tensor = torch.utils.data.TensorDataset(torch.tensor(valid_x.reshape((-1, 1, 32, 32))).float(),
                                                  torch.tensor(np.argmax(valid_y, axis=1)).long())
    valid_loader = DataLoader(valid_tensor, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader, valid_loader


def get_dataset_emnist():
    from torchvision import transforms
    global nr_train
    global nr_valid

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    data = torchvision.datasets.EMNIST(r'../EMNIST', download=True, split='balanced', transform=transform)
    num_train = len(data)
    indices = list(range(num_train))

    nr_train = 100000
    nr_valid = num_train - nr_train

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=nr_valid, replace=False)
    indices = list(set(indices) - set(validation_idx))
    train_idx = np.random.choice(indices, size=nr_train, replace=False)

    train_loader = DataLoader(data, batch_size=batch_size, num_workers=1,
                              sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(data, batch_size=batch_size, num_workers=1,
                              sampler=SubsetRandomSampler(validation_idx))
    return train_loader, valid_loader


def main(get_dataset, net, lr, max_iter=50):
    dl_train, dl_valid = get_dataset()
    opt = optim.SGD(net.parameters(), lr, momentum=0.9)
    net.to(device)
    lt = []
    at = []
    lv = []
    av = []
    iters = range(max_iter)
    loss_train, acc_train = test(net, dl_train, nr_train)
    loss_valid, acc_valid = test(net, dl_valid, nr_valid)
    print(
        "initial \t train_loss: {:.2f} \t train_acc : {}/{} {:.2f}\t test_loss {:.2f}\t test_acc {}/{} {:.2f}".format(
            loss_train, int(acc_train * nr_train), int(nr_train), acc_train, loss_valid,
            int(acc_valid * nr_valid), int(nr_valid), acc_valid))

    for itr in iters:
        train(net, opt, dl_train)
        loss_train, acc_train = test(net, dl_train, nr_train)
        loss_valid, acc_valid = test(net, dl_valid, nr_valid)
        print(
            "itr: {:02d} \t train_loss: {:.2f} \t train_acc : {}/{} {:.2f}\t test_loss {:.2f}\t test_acc {}/{} {:.2f}".format(
                itr, loss_train, int(acc_train * nr_train), int(nr_train), acc_train, loss_valid,
                int(acc_valid * nr_valid), int(nr_valid), acc_valid))
        lt.append(loss_train)
        lv.append(loss_valid)
        at.append(acc_train)
        av.append(acc_valid)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epochs')
    h1 = ax.plot(iters, at, c='r', label='Train')
    h2 = ax.plot(iters, av, c='g', label='Test')
    ax.legend(handles=h1 + h2)
    ax = fig.add_subplot(122)
    ax.set_ylabel('NLL loss')
    ax.set_xlabel('Epochs')
    h1 = ax.plot(iters, lt, c='r', label='Train')
    h2 = ax.plot(iters, lv, c='g', label='Test')
    ax.legend(handles=h1 + h2)
    plt.show()


def q7_1_4(net):
    import os
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
    from q4 import findLetters

    for img in os.listdir('../images'):
        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images', img)))
        bboxes, bw = findLetters(im1)

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

        letters = np.array(
            [str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + [_ for _ in 'abdefghnqrt'])

        plt.imshow(im1)
        colors = 'bgrycmkwbgrcmykw'
        import sys
        for i, bbox_group in enumerate(merged_bbox):
            bbg = sorted(bbox_group, key=lambda x: x[1])
            for j, bbox in enumerate(bbg):
                ymin, xmin, ymax, xmax = bbox

                roi = bw[ymin:ymax + 1, xmin:xmax + 1].astype(np.float)
                roi /= roi.max()
                roi = 1 - roi
                size = 6 * max(ymax - ymin, xmax - xmin) // 5
                dy = (size - ymax + ymin) // 2
                dx = (size - xmax + xmin) // 2
                roi = np.pad(roi, ((dy, dy), (dx, dx)), 'constant', constant_values=1)
                roi = skimage.morphology.erosion(roi)
                roi = skimage.morphology.erosion(roi)
                roi = skimage.transform.resize(roi, (28, 28))

                x = torch.from_numpy(1 - roi.T.reshape((1, 1, 28, 28))).to(device).float()
                y = net(x)
                _, y = y.max(1)
                c = letters[y.item()]

                minr, minc, maxr, maxc = bbox
                rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                    fill=False, edgecolor=colors[i], linewidth=2)
                plt.gca().add_patch(rect)
                rect = matplotlib.patches.Rectangle((xmin - dx, ymin - dy), xmax - xmin + 2 * dx,
                                                    ymax - ymin + 2 * dy, fill=False, edgecolor='w', linewidth=2)
                plt.gca().add_patch(rect)
                plt.text(xmin, ymin - 20, c)
                if j >= 1:
                    ymin2, xmin2, ymax2, xmax2 = bbg[j - 1]
                    if xmin - xmax2 > (xmax - xmin) * 1.5:
                        sys.stdout.write(' ')
                sys.stdout.write(c)
            sys.stdout.write('\n')
            sys.stdout.flush()
        plt.show()


if __name__ == '__main__':
    # q7.1
    # main(get_dataset_nist, Net71(), 5e-3)
    # q7.2
    # main(get_dataset_mnist, Net72(10), 5e-3)
    # q7.3
    # main(get_dataset_nist_conv, Net72(36), 5e-3)
    # q7.4
    net = Net72(47).to(device)
    # main(get_dataset_emnist, net, 5e-3, max_iter=12)
    net.load_state_dict(torch.load('714.pt'))
    q7_1_4(net)
