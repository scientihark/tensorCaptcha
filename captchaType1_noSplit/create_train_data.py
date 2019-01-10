from __future__ import print_function, absolute_import, division
import os
import json
from skimage import io,color
import numpy as np

DATA_DIR = 'data'
DATA_MAP = os.path.join(DATA_DIR, 'captcha.json')
DATA_FULL_DIR = os.path.join(DATA_DIR, 'captcha')
DATA_TRAIN_DIR = os.path.join(DATA_DIR, 'train')
DATA_TRAIN_FILE = os.path.join(DATA_DIR, 'captcha')

# array of tuple of binary image and label
data_x = []
data_y = []

# load image content json file
with open(DATA_MAP) as f:
    image_contents = json.load(f)

# load image and save letters
counter = 0
failCounter = 0
for fname, content in image_contents.iteritems():
    counter += 1

    image = io.imread(os.path.join(DATA_FULL_DIR, fname), as_gray=True)

    data_x.append(image)
    data_y.append(np.int32(content))

    fpath = os.path.join(DATA_TRAIN_DIR, content)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    image_fname = os.path.join(fpath,fname)
    io.imsave(image_fname,image)

# split into train and test data set
train_num = int(len(data_y) * 0.8) # 80%

# save train data
print('saving dataset')
np.savez_compressed(DATA_TRAIN_FILE,
    x_train=data_x[:train_num], y_train=data_y[:train_num],
    x_test=data_x[train_num:], y_test=data_y[train_num:])


