from __future__ import absolute_import, division, print_function

import argparse
from skimage import io
import tensorflow as tf
from model import captcha_classifier

# disable all warnings
import warnings
warnings.filterwarnings('ignore')
# disable tf runtime message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--fname', default='example.png',  help='captcha image path')

def main(argv):
    args = parser.parse_args(argv[1:])
    if args.fname == None:
        print('Capture Image is missing!')
        return None

    # load image
    image = io.imread(args.fname)

    # predict
    content = ''
    probs = [];

    image = image.astype('float32') / 255
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': image},
        y=None,
        num_epochs=1,
        shuffle=False)
    predictions = captcha_classifier.predict(input_fn=predict_input_fn)
    for pred_dict in predictions:
        class_id = pred_dict['classes']
        prob = pred_dict['probabilities'][class_id]
        class_name = str(class_id) 
        #print('predict str: `{}` with probability: {:.3f}%'.format(
               #  class_name, prob * 100))
        content += class_name
        probs.append(prob)

    #print('Captcha: `{}` with confident: `{:.3f}%`'.format(
          #   content, 100*sum(probs)/len(probs)))
    print(content)

if __name__ == '__main__':
    tf.app.run(main)
