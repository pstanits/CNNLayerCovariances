__author__ = 'pstanitsas'

import numpy as np
import scipy.io as sio
# import sklearn
import matplotlib.pyplot as plt

caffe_root = '/home/isi/pstanitsas/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys

sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, caffe_root + 'python/caffe')
import caffe


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    print(data.shape)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data)


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_quick.prototxt',
                caffe_root + 'examples/cifar10/cifar10_quick_iter_4000.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# print(net.blobs['data'].data.shape)
transformer.set_transpose('data', (2, 0, 1))
blb1 = caffe.proto.caffe_pb2.BlobProto()
dt1 = open(caffe_root + 'examples/cifar10/mean.binaryproto', 'rb').read()
blb1.ParseFromString(dt1)
arr1 = np.array(caffe.io.blobproto_to_array(blb1))
out1 = arr1[0]

transformer.set_mean('data', arr1[0].mean(1).mean(1))  # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data',
                             (2, 1, 0))  # the reference model has channels in (2, 1, 0) BGR order instead of RGB
# set net to batch size of 50
net.blobs['data'].reshape(1, 3, 32, 32)
Layer1 = np.empty((10000, 32, 32))
Layer2 = np.empty((10000, 32, 32))
Layer3 = np.empty((10000, 64, 64))
for h in range(1, 10001):
    filename = 'ProcessedIm/Image' + str(h) + '.png'
    print "ITERATION: ", h
    net.blobs['data'].data[...] = transformer.preprocess('data', (caffe.io.load_image(filename)))
    out = net.forward()
    # plt.figure()
    # plt.imshow(image)

    # for k, v in net.blobs.items():
    #    print k, v.data.shape

    # for k, v in net.params.items():
    #    print k, v[0].data.shape

    filters = net.params['conv1'][0].data
    # vis_square(filters.transpose(0, 2, 3, 1))
    # plt.show()
    feat = net.blobs['conv1'].data[0, :]
    inter_Data = np.empty([feat.shape[0], feat.shape[1] * feat.shape[2]])
    rr = np.empty([feat.shape[1] * feat.shape[2], 1])
    for i in range(0, feat.shape[0]):
        rr = feat[i, :, :].flatten()
        inter_Data[i, :] = np.copy(rr)

    Cov1 = np.cov(inter_Data)
    Layer1[(h - 1), :, :] = Cov1
    # print "Covariance 1 is:", Cov1, Cov1.shape
    # Flatten and print to csv
    # vis_square(feat, padval=1)
    # plt.show()

    filters = net.params['conv2'][0].data
    # vis_square(filters[:32].reshape(32 ** 2, 5, 5))
    # plt.show()
    feat = net.blobs['conv2'].data[0, 0::]
    inter_Data = np.empty([feat.shape[0], feat.shape[1] * feat.shape[2]])
    rr = np.empty([feat.shape[1] * feat.shape[2], 1])
    for i in range(0, feat.shape[0]):
        rr = feat[i, :, :].flatten()
        inter_Data[i, :] = np.copy(rr)
    Cov2 = np.cov(inter_Data)
    Layer2[(h - 1), :, :] = Cov2
    # print "Covariance 2 is:", Cov2, Cov2.shape
    # vis_square(feat, padval=0.5)
    # plt.show()

    filters = net.params['conv3'][0].data
    # print(filters.size)
    # vis_square(filters[:32].reshape(32 ** 2, 5, 5))
    # plt.show()

    feat = net.blobs['conv3'].data[0, 0::]
    inter_Data = np.empty([feat.shape[0], feat.shape[1] * feat.shape[2]])
    rr = np.empty([feat.shape[1] * feat.shape[2], 1])
    for i in range(0, feat.shape[0]):
        rr = feat[i, :, :].flatten()
        inter_Data[i, :] = np.copy(rr)
    Cov3 = np.cov(inter_Data)
    Layer3[(h - 1), :, :] = Cov3
    # print "Covariance 3 is:", Cov3, Cov3.shape
    # vis_square(feat, padval=0.5)
    # plt.show()


sio.savemat('Covariance1.mat', mdict={'Layer1': Layer1})
sio.savemat('Covariance2.mat', mdict={'Layer2': Layer2})
sio.savemat('Covariance3.mat', mdict={'Layer3': Layer3})
