import numpy as np
import scipy.ndimage
import os, time
import skimage.io
import skimage.color
import skimage.transform
import util

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H,W,3)
    * vgg16_weights: numpy.ndarray of shape (L,3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''
    image = x
    # change image indo float32 format
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = skimage.color.rgba2rgb(image)
    if image.dtype==np.uint8:
        image = image.astype(np.float32)/255.0
    assert(image.max()<=1.0 and image.min()>=0.0)
    # resize and norm
    image = skimage.transform.resize(image, (224,224,3))
    image -= np.array([[[0.485,0.456,0.406]]]) # mean
    image /= np.array([[[0.229,0.224,0.225]]]) # std

    x = image
    for info in vgg16_weights[:31]: # feature
        print (x.shape)
        name = info[0]
        if name=='conv2d':
            x = multichannel_conv2d(x, info[1], info[2])
        elif name=='relu':
            x = relu(x)
        elif name=='maxpool2d':
            x = max_pool2d(x, info[1])
        elif name=='linear':
            x = linear(x, info[1], info[2])
    print (x.shape)
    x = x.transpose((2,0,1)).flatten()
    for info in vgg16_weights[31:34]: # linear relu linear
        print (x.shape)
        name = info[0]
        if name=='conv2d':
            x = multichannel_conv2d(x, info[1], info[2])
        elif name=='relu':
            x = relu(x)
        elif name=='maxpool2d':
            x = max_pool2d(x, info[1])
        elif name=='linear':
            x = linear(x, info[1], info[2])
    print (x.shape)
    return x


def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H,W,input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H,W,output_dim)
    '''
    h, w, c_in = x.shape
    assert (weight.shape[1] == c_in)
    c_out, c_in, kernel_size, _ = weight.shape
    assert (weight.shape[2] == weight.shape[3])
    assert (bias.shape[0] == c_out)

    xw = np.zeros((h, w, c_out))
    for j in range(c_out):
        for i in range(c_in):
            xw[:, :, j] += scipy.ndimage.convolve(
                x[:, :, i], weight[j, i, ::-1, ::-1], mode='constant', cval=0.0)  # h, w
    xw += bias
    return xw


def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''
    return np.maximum(x, 0)


def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H,W,input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size,W/size,input_dim)
    '''
    xv, yv = np.meshgrid(np.arange(size), np.arange(size))
    pool = [x[i::size, j::size, :] for i, j in zip(xv.flatten(), yv.flatten())]
    r = pool[-1]
    for p in pool:
        r = np.maximum(r, p[:r.shape[0], :r.shape[1], :])
    return r


def linear(x, W, b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''
    return np.einsum('i,ji->j', x, W) + b


if __name__ == '__main__':
    # test these fucing functions
    import torch.nn.functional as F
    import torch
    import torchvision

    # test modules
    if False:
        x = np.random.randn(224, 224, 3)
        weight = np.random.randn(64, 3, 3, 3)
        bias = np.random.randn(64, )

        conv_torch = F.conv2d(
            torch.from_numpy(x[None, :, :, :].transpose((0, 3, 1, 2))).float(),
            torch.from_numpy(weight).float(),
            bias=torch.from_numpy(bias).float(),
            stride=(1, 1),
            padding=(1, 1)
        ).numpy()[0].transpose((1, 2, 0))
        conv_test = multichannel_conv2d(x, weight, bias)
        assert (np.allclose(conv_test, conv_torch, rtol=1e-3, atol=1e-3))

        relu_torch = F.relu(torch.from_numpy(x)).numpy()
        relu_test = relu(x)
        assert (np.allclose(relu_test, relu_torch, rtol=1e-3, atol=1e-3))

        x = np.random.randn(375, 125, 3)
        size = 2
        maxpool_torch = F.max_pool2d(
            torch.from_numpy(x[None, :, :, :].transpose((0, 3, 1, 2))),
            2
        ).numpy()[0].transpose((1, 2, 0))
        maxpool_test = max_pool2d(x, size)
        assert (np.allclose(maxpool_test, maxpool_torch, rtol=1e-3, atol=1e-3))

        x = np.random.randn(64)
        w = np.random.randn(3, 64)
        b = np.random.randn(3)
        fc_torch = F.linear(torch.from_numpy(x[None, :]), torch.from_numpy(w), torch.from_numpy(b)).numpy()[0]
        fc_test = linear(x, w, b)
        assert (np.allclose(fc_test, fc_torch, rtol=1e-3, atol=1e-3))
    # test the whole network
    if True:
        path_img = r"..\data\desert\sun_bzpjfetivojhplyc.jpg"
        image = skimage.io.imread(path_img)

        import cProfile, pstats, io

        pr = cProfile.Profile()
        pr.enable()
        if True:
            # change image indo float32 format
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] == 4:
                image = skimage.color.rgba2rgb(image)
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            assert (image.max() <= 1.0 and image.min() >= 0.0)
            # resize and norm
            image = skimage.transform.resize(image, (224, 224, 3))
            image -= np.array([[[0.485, 0.456, 0.406]]])  # mean
            image /= np.array([[[0.229, 0.224, 0.225]]])  # std
            image = torch.from_numpy(image[None, :, :].transpose(0, 3, 1, 2)).float()
            vgg16 = torchvision.models.vgg16(pretrained=True)
            vgg16.eval()
            vgg_torch = vgg16.features(image)
            vgg_torch = vgg_torch.view((-1, 25088))
            for i in [0,1,3]:
                layer = vgg16.classifier[i]
                vgg_torch = layer(vgg_torch)
            vgg_torch = vgg_torch.detach().numpy()[0]
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(.2)
        print(s.getvalue())

        image = skimage.io.imread(path_img)
        pr = cProfile.Profile()
        pr.enable()
        if True:
            vgg_test = extract_deep_feature(image, util.get_VGG16_weights())
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(.2)
        print(s.getvalue())

        assert  (np.allclose(vgg_torch, vgg_test, 1e-3, 1e-3))
