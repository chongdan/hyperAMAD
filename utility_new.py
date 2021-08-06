"""
Created on 2019-04-29
@author: hty
"""
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from attention_t import attention
from tensorflow.keras.layers import add
from detection import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('deepth', 224,
                            """deepth""")
tf.app.flags.DEFINE_integer('NUM_EPOCHES', 50,
                            """iteration epochs""")
tf.app.flags.DEFINE_integer('batchsize', 64,
                            """Number of images to process in a batch""")
tf.app.flags.DEFINE_integer('NUM_EPOCHS_PER_DECAY', 300,
                            """Number of epochs to decay the learning rate""")
tf.app.flags.DEFINE_float('LEARNING_RATE_DECAY_FACTOR', 0.1,
                          """learning rate decay speed""")
tf.app.flags.DEFINE_float('INITIAL_LEARNING_RATE', 1e-2,
                          """initial learning rate""")
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,
                          """average the weights""")
tf.app.flags.DEFINE_float('scale', 3,
                          """sample scale""")
dropout_rate = 0.5
classes = 2

def _variable_on_gpu(name, shape, initializer):
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var
def weight_variable(name, shape, stddev, wd=0.0005):
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

    
    
def bias_variable(name,shape):
    biases = _variable_on_gpu(name, shape, tf.constant_initializer(0.0))
    return biases
    
    
def relu(x):
    return tf.nn.relu(x)
    
    

def Concatenation(layers):

    return tf.concat(layers,axis=4)

def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def loss(logpros, labels, train=True):
    if train:
        labels = tf.reshape(labels, [FLAGS.batchsize])
    else:
        labels = tf.reshape(labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logpros, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op




def conv3d(x, shape, name, stride=[1, 1, 1, 1, 1],pad='SAME',  use_bias=True):
    w_name = name + '_w'
    b_name = name + '_b'
    weight = weight_variable( w_name, shape, stddev=0.05)
    
    y = tf.nn.conv3d(x, weight, strides=stride, padding=pad)
    if use_bias is True:
        bias = bias_variable(b_name, shape[4])
        y = y + bias


    #tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
    return y


def conv3d_f(x, shape, name, stride=[1, 1, 1, 1, 1], pad='VALID', use_bias=True):
    w_name = name + '_w'
    b_name = name + '_b'
    weight = weight_variable(w_name, shape,stddev=0.05)

    y = tf.nn.conv3d(x, weight, strides=stride, padding=pad)
    if use_bias is True:
        bias = bias_variable(b_name, shape[4])
        y = y + bias

    # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight)
    return y


def deconvolution(x, scale, input_channels, padding='SAME'):
    scale = int(scale)
    output_shape = [ x.shape[0].value,  scale * x.shape[1].value, scale * x.shape[2].value, x.shape[3].value, input_channels]
    stride = [1, scale, scale, 1,  1]

    weight = weight_variable(shape=[3, 3, 1, 1, input_channels], name='weight_reconstruction', stddev=0.05)
    bias = bias_variable(shape=output_shape[-1], name='bias_reconstruction')

    output = tf.nn.conv3d_transpose(x, weight, output_shape, stride, padding=padding) + bias

    return output

def compute_53(temp):
    tensor_0 = temp[:, 0:1, :, :, :]
    tensor_1 = temp[:, 4:5, :, :, :, ]
    tensor_5 = tf.concat([tensor_1, tensor_0], 2)
    tensor_0 = temp[:, 1:4, 0:1, :, :]
    tensor_0 = tf.reshape(tensor_0, (-1, 1, 3, temp.shape[3].value, 20))
    tensor_5 = tf.concat([tensor_5, tensor_0], 2)
    tensor_0 = temp[:, 1:4, 4:5, :, :]
    tensor_0 = tf.reshape(tensor_0, (-1, 1, 3, temp.shape[3].value, 20))
    tensor_5 = tf.concat([tensor_5, tensor_0], 2)
    tensor_1 = temp[:, 2:3, 2:3, :, :]

    tensor_5 = tf.acos((tensor_5 * tensor_1) / (tf.norm(tensor_5) * tf.norm(tensor_1))) * tf.abs(tensor_5 - tensor_1)


    return tensor_5


def FSRCNN(image):

    temp = conv3d(image, shape=[3, 3, 1, 1, 64], name='feature_extraction')
    temp = relu(temp)
    #pool1 = tf.nn.max_pool3d(temp, [1, 1, 1, 3, 1], [1, 1, 1, 3, 1], padding='SAME')
    temp = conv3d(temp, shape=[1, 1, 1, 64, 32],  name='shrinking')
    temp = relu(temp)
    #pool2 = tf.nn.max_pool3d(temp, [1, 1, 1, 3, 1], [1, 1, 1, 3, 1], padding='SAME')
    #temp = dense_net(temp, nb_blocks=2, training_flag=training_flag)
    temp = conv3d(temp, shape=[3, 3, 1, 32, 9], name='mapping1')
    temp = relu(temp)

    temp = conv3d(temp, shape=[1, 1, 1, 9, 9], name='mapping2')
    temp = relu(temp)

   # feature_sum = attention(temp)



   # super_r = deconvolution(feature_sum, FLAGS.scale, feature_sum.shape[-1].value)

    tensor_5 = compute_53(temp)
    temp = conv3d_f(tensor_5, shape=[1, tensor_5.shape[2].value, tensor_5.shape[3].value, tensor_5.shape[-1].value, 40], name='fc_conv1')
    temp = relu(temp)
    temp = tf.nn.dropout(temp, dropout_rate)
    #temp = Drop_out(temp, rate=dropout_rate, training=training_flag)
    temp = conv3d_f(temp, shape=[1, 1, 1, temp.shape[-1].value, 40], name='fc_conv2')
    temp = relu(temp)
    # temp = Drop_out(temp, rate=dropout_rate, training=training_flag)
    scores = conv3d_f(temp, shape=[1, 1, 1, temp.shape[-1].value, classes], name='scores')
    logits_flat = tf.reshape(scores, [-1, classes])
    return logits_flat
