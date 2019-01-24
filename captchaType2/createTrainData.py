from __future__ import print_function, absolute_import, division
import os
import json
from skimage import io,color
from img import split_letters
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
for fname, contents in image_contents.iteritems():
    counter += 1
    image = io.imread(os.path.join(DATA_FULL_DIR, fname))
    
    # split image
    letters = []

    try:
        letters = split_letters(image,debug=True)
    except:
        print('split_letters failed')
        failCounter += 1
    else:
        if letters != None:
            fname = fname.replace('.jpg', '.png')
            print(counter)
            for i, letter in enumerate(letters):
                content = contents[i]
                # add to dataset
                data_x.append(letter)
                data_y.append(np.uint8(ord(content) - 48)) # 48: '0'

                # save letter into train folder
                fpath = os.path.join(DATA_TRAIN_DIR, content)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                letter_fname = os.path.join(fpath, str(i+1) + '-' + fname)
                io.imsave(letter_fname, 255 - letter) # invert black <> white color
        else:
            failCounter += 1
            print('Letters is not valid')
    
    
# split into train and test data set
train_num = int(len(data_y) * 0.8) # 80%

# save train data
print('saving dataset')
np.savez_compressed(DATA_TRAIN_FILE,
    x_train=data_x[:train_num], y_train=data_y[:train_num],
    x_test=data_x[train_num:], y_test=data_y[train_num:])

print(counter)
print(failCounter)
print((counter - failCounter)/counter)

