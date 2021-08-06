# !/usr/bin/env python
# coding: utf-8
import h5py

import scipy.io as sio
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from net import inference, DataSet, loss, _add_loss_summaries
import os
import shutil
FLAGS = tf.app.flags.FLAGS

sess = tf.InteractiveSession()

# import data from mat file
'''
train_dataf = 'traindata.mat'
train_labelf = 'traindata.mat'


#train_data = np.array(h5py.File(train_dataf)['traindata']).T
train_data = sio.loadmat(train_dataf)['train']

train_labels = sio.loadmat(train_labelf)['testlabel']


classes = np.max(train_labels) + 1
channels = train_data.shape[1]
deepth = train_data.shape[2]
height = train_data.shape[3]
width = train_data.shape[4]
'''

def read_data(path):
    with h5py.File(path, 'r') as hf:
        train_data = np.array(hf.get('hr_data'))
        train_data = train_data[:, 5:10, 5:10, 6:-6, :]
        train_label = np.array(hf.get('label'))


        # train_data = np.transpose(data, (0, 2, 3, 1))
        # train_label = np.transpose(label, (0, 2, 3, 1))
        print(train_data.shape)
        print(train_label.shape)
    return train_data, train_label







def trainop(total_loss, global_step,trainset):
    num_batches_per_epoch = trainset.num_examples / FLAGS.batchsize  # 一个epoch的batch num

    decay_steps = int(num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY)  # 一个epoch的学习率需要衰减的次数
    # Decay the learning rate exponentially based on the number of steps.
    # 就是说在每一次迭代过程中，都需要重新计算一次learning rate，
    # 而这里初始的INITIAL_LEARNING_RATE为0，
    # global_step为当前的迭代次数,每训练一个batch就加1，decay_steps就是每多少代，
    # learning_rate衰减到到LEARNING_RATE_DECAY_FACTOR×INITIAL_LEARNING_RATE值，
    # 比如本程序中LEARNING_RATE_DECAY_FACTOR = 0.1 ，而decay_steps = num_batches_per_epoch * NUM_EPOCHS_PER_DECAY = 50000 / 128×350
    # ，也就说每十多万次迭代，lr衰减为原来0.1，然后根据每代的lr，用梯度法计算

    lr = tf.train.exponential_decay(FLAGS.INITIAL_LEARNING_RATE,
                                    global_step,  # 当前迭代轮数
                                    decay_steps,  # 过完所有的训练数据需要的迭代次数
                                    FLAGS.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.MOVING_AVERAGE_DECAY, global_step)  # variable_averages是一个对象,1-moving_average_decay相当于求moving average时的更新率
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())  # 这个对象的apply()函数先创造一个变量的影子,然后对影子训练变量求一个moving average,返回这个op.训练参数的moving average要比最终训练得到的参数效果要好很多.

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(
            name='train')  # 在进行1.梯度更新(即对所有训练参数进行跟新);2.求参数的moving averge后,方可进行tf.no_op()操作;tf.no_op仅仅创造一个操作的占位符

    return train_op

def train(train_file, model_save_path, reuse=None):

    train_data, train_label = read_data(train_file)

    classes = np.max(train_label) + 1
    channels = train_data.shape[4]
    deepth = train_data.shape[3]
    height = train_data.shape[2]
    width = train_data.shape[1]
    trainset = DataSet(train_data, train_label, dtype=tf.float32)
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batchsize,  height, width, deepth, channels))
    label_placeholder = tf.placeholder(tf.int64, shape=(None, 1))

    global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope('inference', reuse=reuse) as scope:
        logits = inference(images_placeholder,3)
    loss_ = loss(logits, label_placeholder, True)
    y_conv = tf.nn.softmax(logits)
    #y_ = tf.slice(y_conv, [0, 1], [-1, classes-1 ])
    train_op = trainop(loss_, global_step, trainset )
    s = 1

    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    matrix = np.zeros((classes, classes))
    saver = tf.train.Saver()
    sess.run(init)
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    #summary_writer = tf.summary.FileWriter('hyper_log', graph_def=graph_def)
    max_step = int((trainset.num_examples / FLAGS.batchsize))
    t = 1
    #max_step = 400
    for step in range(max_step*FLAGS.NUM_EPOCHES):
        image, label = trainset.next_batch(FLAGS.batchsize, True)


        sess.run(train_op, feed_dict={images_placeholder: image, label_placeholder: label})
        if step % 100 == 0:
            # label_placeholder: label})
            summary_str, lv = sess.run([summary_op, loss_], feed_dict={images_placeholder: image,
                                                                       label_placeholder: label})

            prediction_list = []
            predictions = sess.run(tf.argmax(y_conv, 1), feed_dict={images_placeholder: image})
            prediction_list.extend(predictions)
            pre_list = prediction_list


            for i in range(FLAGS.batchsize):
                pre = pre_list[i]
                la = label[i, 0]
                matrix[pre, la] += 1

            print (np.int_(matrix))
            print (np.sum(np.trace(matrix)))  # 返回对角线元素之和
            print ('precision= %f' % (np.sum(np.trace(matrix)) / float((FLAGS.batchsize)*t)))
            t += 1
            print('step %d, loss %f' % (step, lv))
            #summary_writer.add_summary(summary_str, step)

            ep = step // 100
            path = model_save_path + '/save/' + str(ep) + '/'
            save_model = saver.save(sess, path + 'FSRCNN_model')
            new_path = model_save_path + '/' + str(ep) + '/'
            shutil.move(path, new_path)
            '''
            模型首先是被保存在save下面的,直接保存的话，前面的epoch对应的文件夹会出现内部文件被删除的情况，原因不明；所以这里用shutil.move把模型所在的文件夹移动了一下
            '''
            print("\nModel saved in file: %s" % save_model)
            #saver.save(sess, 'checkpoint/model.ckpt', global_step=step)


def main():
    train_file = 'traindata.h5'
    # train_file = path = file_name("./" + database + "/X" + str(scale), ".h5")

    model_save_path = 'FSRCNN_checkpoint189_5_detection'

    if os.path.exists(model_save_path) == False:
        print('The ' + '"' + model_save_path + '"' + 'can not find! Create now!')
        os.mkdir(model_save_path)

    if os.path.exists(model_save_path + '/save') == False:
        print('The ' + '"save' + '"' + ' can not find! Create now!')
        os.mkdir(model_save_path + '/save')

    train(train_file,model_save_path,None)


if __name__ == '__main__':
    main()

