from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import datetime
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import os
import numpy as np


# Define a weight function
def weight_variable(shape):
    # A tensor of the specified shape filled with 0.1 truncated normal values
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Defind a bias function
def bias_variable(shape):
    # A tensor of the specified shape filled with constant 0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Define a convolution function, where x is the inout and W is the kernel.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Define a max pooling function, where x is the result of the convolution
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# def variable_summaries(var):
#     with tf.name_scope("summaries"):
#         # 计算参数的均值，并使用tf.summary.scaler记录
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar("mean", mean)
#
#         # 计算参数的标准差
#         with tf.name_scope("stddev"):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
#         tf.summary.scalar("stddev", stddev)
#         tf.summary.scalar("max", tf.reduce_max(var))
#         tf.summary.scalar("min", tf.reduce_min(var))
#         # 用直方图记录参数的分布
#         tf.summary.histogram("histogram", var)


def conv_layer(kernel_width, kernel_height, input_num, kernel_num, input_tensor, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            W_conv = weight_variable([kernel_width, kernel_height, input_num, kernel_num])  # 4-d tensor
            # variable_summaries(W_conv)
            tf.summary.histogram("W_conv", W_conv)
        with tf.name_scope("biases"):
            b_conv = bias_variable([kernel_num])
            # variable_summaries(b_conv)
            tf.summary.histogram("b_conv", b_conv)
        with tf.name_scope("linear_compute"):
            h_conv = tf.nn.elu(conv2d(input_tensor, W_conv) + b_conv)
            tf.summary.histogram("convolution_layer", h_conv)
        h_pool = max_pool_2x2(h_conv)
        tf.summary.histogram("pooling_layer", h_pool)
        return h_pool


class Config:
    epoch_num = 800  # 整个数据集循环次数
    kernel = [5, 5]  # Kernel Size 5*5
    fc_num = 1024  # 定义隐藏层神经元数量
    class_num = 10  # 定义类别数量
    learn_rate = 0.0001  # 定义学习率
    print_interval = 100  # 每隔多少轮训练输出一次结果
    batch_size = 50  # Define batch size

    log_path = "/Users/apple/Documents/PyCharmProjects/CNNforMNIST2.0/log/log-" + datetime.date.today().__str__() + ".txt"
    summary_path = "/Users/apple/Documents/PyCharmProjects/CNNforMNIST2.0/summary/"


def run():

    config = Config

    log = open(config.log_path, 'a', encoding='utf-8')  # 'a' = append: append the log
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    log.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")

    # Import the MNIST dataset from tensorflow
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
    print("Data Directory:" + FLAGS.data_dir)
    log.write("Data Directory:" + FLAGS.data_dir + "\n")
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    print("CNN Structure: 1 input layer, 2 convolution layers, 1 fully-connected layer, and 1 softmax output layer")
    log.write("CNN Structure: 1 input layer, 2 convolution layers, 1 fully-connected layer, and 1 softmax output layer" + "\n")
    log.flush()

    with tf.name_scope("input_image"):
        x = tf.placeholder(tf.float32, [None, 784], name="x-input")

    with tf.name_scope("input_reshape"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, config.class_num)  # shows 10 images
        print("Input Layer: 1 channel, 28*28 pixels images")
        log.write("Input Layer: 1 channel, 28*28 pixels images" + "\n")
        log.flush()

    with tf.name_scope("conv_layer_1"):
        h_pool1 = conv_layer(config.kernel[0], config.kernel[1], 1, 32, x_image, "conv_layer_1")
        print("The 1st convolution layer:  32 of 5*5 kernels, activation function: elu, pooling: mx_pool_2x2")
        log.write("The 1st convolution layer:  32 of 5*5 kernels, activation function: elu, pooling: mx_pool_2x2" + "\n")
        log.flush()

    with tf.name_scope("conv_layer_2"):
        h_pool2 = conv_layer(config.kernel[0], config.kernel[1], 32, 64, h_pool1, "conv_layer_2")
        print("The 2nd convolution layer:  64 of 5*5 kernels, activation function: elu, pooling: mx_pool_2x2")
        log.write("The 2nd convolution layer:  64 of 5*5 kernels, activation function: elu, pooling: mx_pool_2x2" + "\n")
        log.flush()

    with tf.name_scope("fully_connect_layer"):
        # The input of kernel is 7*7*64, The output is 1024, 2-D Tensor
        W_fc = weight_variable([7 * 7 * 64, config.fc_num])
        # variable_summaries(W_fc)
        tf.summary.histogram("W_fc", W_fc)
        b_fc = bias_variable([config.fc_num])
        # variable_summaries(b_fc)
        tf.summary.histogram("b_fc", b_fc)
        # Flatten the input image into 1-D vector with 7*7*64
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        # tf.matmul -> one-one multiply for two vector
        h_fc = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
        tf.summary.histogram('fully_connect_layer', h_fc)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc, keep_prob)
        print("The fully-connected layer: activation function: elu, dropout: 0.5")
        log.write("The fully-connected layer: activation function: elu, dropout: 0.5" + "\n")
        log.flush()

    with tf.name_scope("softmax"):
        W_sfm = weight_variable([config.fc_num, config.class_num])
        # variable_summaries(W_sfm)
        tf.summary.histogram("W_sfm", W_sfm)
        b_sfm = bias_variable([config.class_num])
        # variable_summaries(b_sfm)
        tf.summary.histogram("b_sfm", b_sfm)
        y_output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_sfm) + b_sfm)
        tf.summary.histogram("y_output", y_output)
        print("The softmax output layer: Softmax Function")
        log.write("The softmax output layer: Softmax Function" + "\n")
        log.flush()

    with tf.name_scope("label"):
        y_ = tf.placeholder(tf.float32, [None, config.class_num])

    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_output), reduction_indices=[1]))
        tf.summary.scalar("cross_entropy", cross_entropy)
        print("The error function: The cross-entropy cost function")
        log.write("The error function: The cross-entropy cost function" + "\n")
        log.flush()

    with tf.name_scope("train"):
        # Replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
        train_step = tf.train.AdamOptimizer(config.learn_rate).minimize(cross_entropy)
        print("The training algorithm: adam algorithm with learning rate 0.0001")
        log.write("The training algorithm: adam algorithm with learning rate 0.0001" + "\n")
        log.flush()

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.histogram("accuracy", accuracy)
        tf.summary.scalar("accuracy", accuracy)

    summary_merge = tf.summary.merge_all()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(config.summary_path, session.graph)

    for i in range(config.epoch_num):
        batch_x, batch_y = mnist.train.next_batch(config.batch_size)
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
        if i % config.print_interval == 0:
            # eval means run
            # Using feed_dict to replace the placeholder tensors x, y_, and keep_prob with the training examples.
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            print("Accuracy of the %d training cycle: %g" % (i, train_accuracy))
            log.write("Accuracy of the %d training cycle: %g" % (i, train_accuracy) + "\n")
            log.flush()
            result_summary = session.run(summary_merge, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            writer.add_summary(result_summary, i)

    print("Accuracy of testing: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    log.write("Accuracy of testing: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) + "\n")
    log.flush()

    # Embedding Projector Visualization
    meta_file = "metadata.tsv"
    embedding_var = tf.Variable(batch_x, name="trainimages") # Create the embedding
    summary_writer = tf.summary.FileWriter(config.summary_path)
    projector_config = projector.ProjectorConfig()
    embed = projector_config.embeddings.add()
    embed.tensor_name = embedding_var.name
    embed.metadata_path = meta_file
    projector.visualize_embeddings(summary_writer, projector_config)

    # Create and initiate the session to store
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(config.summary_path, 'metadata.ckpt'))

    # Create the metadata.tsv
    with open(config.summary_path+meta_file, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(batch_y):
            print(index)
            inx, = np.where(label == 1)
            print(inx)
            f.write("%d\t%d\n" % (index, inx))
        f.close()

    writer.close()

run()







