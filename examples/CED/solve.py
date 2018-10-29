from __future__ import division
import numpy as np
import sys
caffe_root = '/path/to/caffe_root' 
sys.path.insert(0, caffe_root + '/python')
import caffe


base_weights = './hed_vgg16.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device()

solver = caffe.SGDSolver('solver.prototxt')

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

solver.step()