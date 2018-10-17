import numpy as np
import scipy.ndimage

import cProfile, pstats, io

x = np.random.randn(256, 3, 224, 224)
weight = np.random.randn(4, 3, 3, 3)


def fast_conv(x, weight):
    '''
    fast convolution using im2col
    :param x: (n, c_in, ih, iw) or (c_in, ih, iw) or (ih, iw)
    :param weight: (c_out, c_in, h, w) or (c_in, h, w) or (h, w)
    :return: convolution result
    '''

    # step O: add axis
    if len(x.shape) == 2:
        c_in = 1
        n = 1
        ih, iw = x.shape
    elif len(x.shape) == 3:
        n = 1
        c_in, ih, iw = x.shape
    elif len(x.shape) == 4:
        n, c_in, ih, iw = x.shape
    x.shape = n, c_in, ih, iw

    if len(weight.shape) == 2:
        assert (c_in == 1)
        c_out = 1
        h, w = weight.shape
    elif len(weight.shape) == 3:
        assert (c_in == weight.shape[0])
        c_out = 1
        c_in, h, w = weight.shape
    elif len(weight.shape) == 4:
        assert (c_in == weight.shape[1])
        c_out, c_in, h, w = weight.shape
    weight.shape = c_out, c_in, h, w
    assert ((h - 1) % 2 == 0)
    assert ((w - 1) % 2 == 0)
    dh = (h - 1) // 2
    dw = (w - 1) // 2

    # step I: im2col
    x_pad = np.pad(x, ((0, 0), (0, 0), (dh, dh), (dw, dw)), mode='constant')
    col = np.zeros((n, ih, iw, c_in * h * w))

    for idy in range(ih):
        for idx in range(iw):
            cy, cx = idy + dh, idx + dw
            col[:, idy, idx, :] = x_pad[:, :, cy - dh:cy + dh + 1, cx - dw:cx + dw + 1].reshape((n, c_in * h * w))

    # conv = np.einsum('nihwk,oik->nohw', col, weight.reshape((c_out, c_in, h*w)))
    conv = col.reshape((n * ih * iw, c_in * h * w)).dot(weight.reshape((c_out, c_in * h * w)).transpose())
    conv = conv.reshape((n, ih, iw, c_out)).transpose((0, 3, 1, 2))
    return conv


pr = cProfile.Profile()
pr.enable()
r_fast_conv = fast_conv(x, weight)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(.2)
print(s.getvalue())


def multichannel_conv2d(x, weight):
    n, c_in, h, w = x.shape
    assert (weight.shape[1] == c_in)
    c_out, c_in, kernel_size, _ = weight.shape
    assert (weight.shape[2] == weight.shape[3])

    xw = np.zeros((n, c_out, h, w))
    for k in range(n):
        for j in range(c_out):
            for i in range(c_in):
                xw[k, j, :, :] += scipy.ndimage.convolve(
                    x[k, i, :, :], weight[j, i, ::-1, ::-1], mode='constant', cval=0.0)  # h, w
    return xw


pr = cProfile.Profile()
pr.enable()
r_scipy_conv = multichannel_conv2d(x, weight)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(.2)
print(s.getvalue())

assert (np.allclose(r_scipy_conv, r_fast_conv, 1e-3, 1e-3))
