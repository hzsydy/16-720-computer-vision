import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random
import math
from params import *

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    # change image indo float32 format
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = skimage.color.rgba2rgb(image)
    if image.dtype==np.uint8:
        image = image.astype(np.float32)/255.0
    assert(image.max()<=1.0 and image.min()>=0.0)
    image = skimage.color.rgb2lab(image)

    h, w, _ = image.shape
    result = np.zeros((h, w, 60))
    for sidx, scale in enumerate([1, 2, 4, 8, 8*math.sqrt(2)]):
        for cidx in range(3):
            gaussian = scipy.ndimage.filters.gaussian_filter(image[:, :, cidx], sigma=scale)
            laplacian = scipy.ndimage.laplace(gaussian)
            derivative_x = scipy.ndimage.prewitt(gaussian, axis=1)
            derivative_y = scipy.ndimage.prewitt(gaussian, axis=0)
            result[:, :, cidx+3*(4*sidx+0)] = gaussian
            result[:, :, cidx+3*(4*sidx+1)] = laplacian
            result[:, :, cidx+3*(4*sidx+2)] = derivative_x
            result[:, :, cidx+3*(4*sidx+3)] = derivative_y
    return result





def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    filter_response = extract_filter_responses(image)
    h, w, c = filter_response.shape
    features = filter_response.reshape((h*w, c))
    dist = scipy.spatial.distance.cdist(features, dictionary)
    wordmap = np.argmin(dist, axis=1)
    wordmap.shape = h, w
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''


    i,alpha,image_path = args
    feature_name = '../temp/sampled_feature_{:d}.npy'.format(i)
    if os.path.isfile(feature_name):
        #print ('File already exist', feature_name)
        return
    rgb_image = scipy.ndimage.imread(image_path)
    filter_response = extract_filter_responses(rgb_image)
    h, w, c = filter_response.shape
    # randomly choose alpha pixels for output
    features = filter_response.reshape((h*w, c))
    sampled_features = features[np.random.permutation(np.arange(h*w))[:alpha], :]
    np.save(feature_name, sampled_features)
    return


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    try:
        os.mkdir('../temp')
    except FileExistsError:
        pass
    print('computing features...')
    train_data = np.load("../data/train_data.npz")
    pool = multiprocessing.Pool(processes=num_workers)
    T = len(train_data['image_names'])
    args = [(idx, alpha, os.path.join('../data',str(fn[0]))) for idx, fn in enumerate(train_data['image_names'])]
    pool.map(compute_dictionary_one_image, args)

    print('building dictionary...')
    all_features = np.zeros((alpha*T, 60))
    for idx, _, _ in args:
        all_features[alpha*idx:alpha*idx+alpha, :] = np.load('../temp/sampled_feature_{:d}.npy'.format(idx))
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(all_features)
    #kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=K, batch_size=128).fit(all_features)
    dictionary = kmeans.cluster_centers_
    np.save('dictionary.npy', dictionary)
    return




