import numpy as np
import threading
import queue
import imageio
import os, time
import math
import scipy
import visual_words
from params import *
import multiprocessing


def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    print ('extracting features from training set images...')
    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    SPM_layer_num = 3
    pool = multiprocessing.Pool(processes=num_workers)
    results = [pool.apply_async(get_image_feature, args=(fn, dictionary, SPM_layer_num-1, K))
               for fn in [os.path.join('../data', str(f[0])) for f in train_data['image_names']]]
    features = np.stack([p.get() for p in results], axis=0)
    labels = train_data['labels']
    np.savez('trained_system.npz', dictionary=dictionary, features=features, labels=labels, SPM_layer_num=SPM_layer_num)


def _predict_label(fn, features, labels, dictionary, SPM_layer_num, K):
    feature = get_image_feature(fn, dictionary, SPM_layer_num - 1, K)
    pd_idx = distance_to_set(feature, features)
    return pd_idx


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    dictionary = trained_system['dictionary']
    features = trained_system['features']
    labels = trained_system['labels']
    SPM_layer_num = int(trained_system['SPM_layer_num'])

    #pd_labels = [_predict_label(fn, features, labels, dictionary, SPM_layer_num, K)
    #            for fn in [os.path.join('../data', str(f[0])) for f in test_data['image_names']]]
    pool = multiprocessing.Pool(processes=num_workers)
    results = [pool.apply_async(_predict_label, args=(fn, features, labels, dictionary, SPM_layer_num, K))
               for fn in [os.path.join('../data', str(f[0])) for f in test_data['image_names']]]
    pd_label_idxs = [p.get() for p in results]
    pd_labels = [labels[pd_idx] for pd_idx in pd_label_idxs]
    gt_labels = test_data['labels']

    confusion = np.zeros((8, 8))
    for gt_label, pd_label in zip(gt_labels, pd_labels):
        confusion[gt_label][pd_label] += 1

    accuracy = np.diag(confusion).sum()/confusion.sum()

    return confusion, accuracy



def get_image_feature(file_path, dictionary, layer_num, _K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    image = scipy.ndimage.imread(file_path)
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, _K)
    return feature


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    word_hist = word_hist.reshape((1, -1))
    hist_similarity = np.minimum(word_hist, histograms).sum(axis=1)
    return np.argmax(hist_similarity)


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    hist = np.histogram(wordmap.flatten(), bins=dict_size, density=True)
    # import matplotlib.pyplot as plt
    # plt.bar(range(K), hist[0])  # arguments are passed to np.histogram
    # plt.title("Feature histogram - bar plot")
    # plt.show()
    return hist[0]


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # the smallest part
    scale = 2 ** layer_num
    h, w = wordmap.shape
    h_feature_small, w_feature_small = math.ceil(h / scale), math.ceil(w / scale)

    feature_small = np.zeros((scale, scale, dict_size))
    for i in range(scale):
        for j in range(scale):
            feature_small[i, j, :] = get_feature_from_wordmap(
                wordmap[i * h_feature_small:(i + 1) * h_feature_small, j * w_feature_small:(j + 1) * w_feature_small],
                dict_size)

    features = [feature_small]
    for log_scale in range(layer_num - 1, -1, -1):
        last_feature = features[-1]
        new_feature = (
                    last_feature[0::2, 0::2] + last_feature[1::2, 0::2] +
                    last_feature[0::2, 1::2] + last_feature[1::2, 1::2]
        ) / 4
        features.append(last_feature)

    weight = [2**(-layer_num-1+i) for i in range(layer_num)]
    weight[0] *= 2

    flatten_feature = [features[i].flatten()*weight[i] for i in range(layer_num)]
    flatten_feature = np.concatenate(flatten_feature, axis=0)
    flatten_feature /= flatten_feature.sum()
    return flatten_feature
