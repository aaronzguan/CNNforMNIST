import tensorflow as tf

# Define a weight function
# A tensor of the specified shape filled with 0.1 truncated normal values
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Define a bias function
# A tensor of the specified shape filled with constant 0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Define a convolution function, where x is the input and W is the kernel.
# conv2d: [input, filter, strides, padding] where
# [input] should follow: [batch_size, in_height, in_width, in_channels]
# [filter] should follow: [filter_height, filter_width, input channel, output channels]
# Our convolution uses a [stride] of one and zero padded so that the output is the same size as the input
# [padding] = SAME -> new_height = new_width = x / Stride, so here strides 1, which means input size = output size
# [padding] = VALID -> Consider the size of kernel(filter), new_height = new_width = (x - W + 1) / Stride
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Define a max pooling function, where x is the result of the convolution and the input of pooling layer
# tf.nn.max_pool is for max pooling while avg_pool is for average pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')