import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os, time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import multiprocessing

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class VGGFeature(nn.Module):
    def __init__(self, vgg):
        super(VGGFeature, self).__init__()
        self.extractor = nn.Sequential(
            vgg.features,
            Flatten(),
            vgg.classifier[0],
            vgg.classifier[1],
            vgg.classifier[3],
        )

    def forward(self, x):
        return self.extractor(x)



def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,K)
    * labels: numpy.ndarray of shape (N)
    '''
    print ('build_recognition_system')
    train_data = np.load("../data/train_data.npz")
    extractor = VGGFeature(vgg16)

    # features = [get_image_feature((i, f, extractor))
    #             for i, f in enumerate(train_data['image_names'][:5])]
    pool = multiprocessing.Pool(processes=num_workers)
    results = [pool.apply_async(get_image_feature, args=((i, f, extractor),))
               for i, f in enumerate(train_data['image_names'])]
    features = [p.get() for p in results]
    features = np.stack(features, axis=0)
    labels = train_data['labels']
    np.savez('trained_system_deep.npz', features=features, labels=labels)


# Feed the data to the model


def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    print ('evaluate_recognition_system')

    test_data = np.load("../data/test_data.npz")
    extractor = VGGFeature(vgg16)

    trained_system = np.load("trained_system_deep.npz")
    train_features = trained_system['features']
    labels = trained_system['labels']

    pool = multiprocessing.Pool(processes=num_workers)
    results = [pool.apply_async(get_image_feature, args=((i, f, extractor),))
               for i, f in enumerate(test_data['image_names'])]
    features = [p.get() for p in results]

    pd_labels = [labels[distance_to_set(f, train_features)] for f in features]
    gt_labels = test_data['labels']

    confusion = np.zeros((8, 8))
    for gt_label, pd_label in zip(gt_labels, pd_labels):
        confusion[gt_label][pd_label] += 1

    return confusion


def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H,W,3)

    [output]
    * image_processed: torch.Tensor of shape (3,H,W)
    '''
    # change image indo float32 format
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = skimage.color.rgba2rgb(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    assert (image.max() <= 1.0 and image.min() >= 0.0)
    # resize and norm
    image_processed = skimage.transform.resize(image, (224, 224, 3))
    image_processed -= np.array([[[0.485, 0.456, 0.406]]])  # mean
    image_processed /= np.array([[[0.229, 0.224, 0.225]]])  # std
    image_processed = torch.from_numpy(image_processed.transpose((2, 0, 1)))
    return image_processed


def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    * time_start: time stamp of start time
    [saved]
    * feat: evaluated deep feature
    '''
    i, image_path, vgg16 = args
    print ('processing image', i, 'at', image_path)
    image_path = os.path.join('../data', str(image_path[0]))
    image = skimage.io.imread(image_path)
    image_processed = preprocess_image(image)
    return vgg16(image_processed[None,:,:,:]).detach().numpy()[0]


def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N,K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''
    feature = feature.reshape((1, -1))
    errors = ((train_features-feature)**2).sum(axis=1)
    return np.argmin(errors)
