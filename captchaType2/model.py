from __future__ import print_function, absolute_import, division
import tensorflow as tf

MODEL_LOG_DIR = 'checkpoint'

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  
  input_layer = tf.reshape(features['x'], [-1, 16, 14,1])
  #print(input_layer)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding='same',
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding='same',
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3 and Pooling Layer #3
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding='same',
      activation=tf.nn.relu)

  # Dense Layer
  conv3_flat = tf.layers.flatten(inputs=conv3)
  dense1 = tf.layers.dense(inputs=conv3_flat, units=512, activation=tf.nn.relu)

  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)

  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout2, units=75) # 74  =>  0-9A-Za-z  48 = '0' 122 = 'z' 

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      'classes': tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predictions['classes'])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

captcha_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=MODEL_LOG_DIR
)
