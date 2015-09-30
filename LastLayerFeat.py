__author__ = 'pstanitsas'

import numpy as np
import scipy.io as sio
# import sklearn
import matplotlib.pyplot as plt

caffe_root = '/scratch/coreprocessing/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys

print sys.path.insert(0, caffe_root + 'python')
print sys.path.insert(0, caffe_root + 'python/caffe')
import caffe

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_full_deploy.prototxt',
                caffe_root + 'examples/cifar10/cifar10_full_iter_70000.caffemodel',
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
LastLayer = np.empty((10000, 10))
for h in range(1, 10001):
    filename = '/home/dimitrios/Desktop/CIFAR_Images/testbatch/Image_' + str(h) + '.png'
    print "ITERATION: ", h
    net.blobs['data'].data[...] = transformer.preprocess('data', (caffe.io.load_image(filename)))
    out = net.forward()
    feat = net.blobs['ip1'].data[0, :]
    print feat.shape
    LastLayer[(h - 1), :] = feat

sio.savemat('LastLayer.mat', mdict={'LastLayer': LastLayer})

