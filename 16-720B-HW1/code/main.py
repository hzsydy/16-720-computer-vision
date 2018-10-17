import numpy as np
import util
from matplotlib import cm
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import torchvision#
import skimage.io
from params import *

if __name__ == '__main__':
    num_cores = 8
    #image = image.astype('float') / 255

    #filter_responses = visual_words.extract_filter_responses(image)
    #util.display_filter_responses(filter_responses)

    #visual_words.compute_dictionary(num_workers=num_cores)

    #dictionary = np.load('dictionary.npy')
    #fig = plt.figure(1)
    #for i, path_img in enumerate([
    #    "../data/kitchen/sun_aasmevtpkslccptd.jpg",
    #    "../data/kitchen/sun_aawefvrbscajixha.jpg",
    #    "../data/kitchen/sun_aaslbwtcdcwjukuo.jpg",
    #    "../data/waterfall/sun_aastyysdvtnkdcvt.jpg",
    #    "../data/waterfall/sun_abcxnrzizjgcwkdn.jpg",
    #    "../data/waterfall/sun_abiyxzwgkjnuroap.jpg",
    # ]):
    #    image = skimage.io.imread(path_img)
    #    image = image.astype('float') / 255
    #    wordmap = visual_words.get_visual_words(image, dictionary)
    #    plt.subplot(3, 4, 4*(i%3)+2*(i//3) + 1)
    #    plt.imshow(wordmap, cmap='#')
    #    plt.axis('off')
    #    plt.subplot(3, 4, 4*(i%3)+2*(i//3) + 2)
    #    plt.imshow(image)
    #    plt.axis('off')
    #plt.show()
    #wordmap = visual_words.get_visual_words(image, dictionary)
    #visual_recog.get_feature_from_wordmap(wordmap, K)
    #util.save_wordmap(wordmap, filename)
    #visual_recog.build_recognition_system(num_workers=num_cores)

    conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    #deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
    conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

