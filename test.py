# y = softmax(Wx+b)
# Define W, x, and b
# x is the image, y is the "labels"
# Each image in MNIST has a corresponding label, a number between 0 and 9 representing the digit drawn in the image.
# Hence, there would be only 10 possible things that a given image can be.
# For this implementation, we we're going to want our labels as "one-hot vectors".
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.
# In this case, the nth digit will be represented as a vector which is 1 in the nth dimension.
# 3 = [0 0 0 1 0 0 0 0 0 0]

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
print("Data Directory:"+FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


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
    # conv2d: [input, filter, strides, padding] where
    # input should follow: [batch_size, in_height, in_width, in_channels]
    # filter should follow: [filter_height, filter_width, input channel, output channels]
    # Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input
    # padding = SAME -> new_height = new_width = x / Stride, so here strides 1, which means input size = output size
    # padding = VALID -> Consider the size of kernel(filter), new_height = new_width = (x - W + 1) / S

# Define a max pooling function, where x is the result of the convolution
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # tf.nn.max_pool is for max pooling while avg_pool is for average pooling


print("Structure: One input layer, 2 convolutional layers and 2 fully-connected layers")
session = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
# Placeholder: When we ask TensorFlow to run a computation, a value that we'll input.
# We wanna input any number of MNIST images, each flattened into a 784-dimensional vector
# 784: the dimensionality of a single flattened 28 by 28 pixel MNIST image
# We represent this as a 2-D tensor of floating-point numbers
# None indicates that the first dimension, corresponding to the batch size, can be of any size.


# Define the input image
x_image = tf.reshape(x, [-1, 28, 28, 1])
# [batch_size, in_height, in_width, in_channels]
# Reshape x to a 4d tensor with the second and third dimensions corresponding to image width and height,
# and the final dimension corresponding to the number of color channels.
# Batch_size: Size of the subset of examples to use when performing gradient descent during training.
# -1 means it will automatically compute based on the input
print("The size of input image: 28 for width, 28 for height, 1 channel")

######Define the first convolutional layer######
# Define the kernel
W_conv1 = weight_variable([5, 5, 1, 32])
# [kernel size, input channel, output channel]
# There should be 32 kernels with 5x5 size
# Define the bias
# Each output channel should have a bias
b_conv1 = bias_variable([32])
# Define the activation function
# Use ReLu Function
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1)+b_conv1)
# Define the pooling
h_pool1 = max_pool_2x2(h_conv1)
print("The first convolutional layer: The numbers of kernel is 32, The size of kernel is 5*5, The activation function is elu, the pooling is mx_pool_2x2")

######Define the Second convolutional layer######
# Define the second convolutional layer
# Define the kernel
W_conv2 = weight_variable([5, 5, 32, 64])
# Define the bias
b_conv2 = bias_variable([64])
# Define the activation function
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
# Define the pooling
h_pool2 = max_pool_2x2(h_conv2)
print("The second convolutional layer: The numbers of kernel is 64, The size of kernel is 5*5, The activation function is elu, the pooling is mx_pool_2x2")


######Define the Third Fully-Connected layer######
# The input of kernel is 7*7*64, The output is 1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# Flatten the input image into 1-D vector with 7*7*64
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# tf.matmul -> one-one multiply for two vector
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# To reduce overfitting, we will apply dropout before the readout layer
# We create a placeholder for the probability that a neuron's output is kept during dropout.
# This allows us to turn dropout on during training, and turn it off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print("The third fully-connected layer: The activation function is elu, the dropout is 0.5")


######Define the Fourth layer######
# The input is 1024, output is 10 (There are 10 labels for output classes)
W_fc2 = weight_variable([1024, 10])
# Define the bias: b is a 10-dimensional vector (because we have 10 classes)
b_fc2 = bias_variable([10])
# Use the Softmax as activation function
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# Define the target output classes
y_ = tf.placeholder(tf.float32, [None, 10])
# The target output classes y_ will also consist of a 2d tensor,
# where each row is a one-hot 10-dimensional vector indicating which
# digit class (zero through nine) the corresponding MNIST image belongs to.
print("The fourth fully-connected layer: Softmax Function")


# Define the error function / Loss function
# Loss indicates how bad the model's prediction was on a single example;
# We try to minimize that while training across all the examples
# Our loss function is the cross-entropy between the target and the softmax activation function applied to the model's prediction
# tf.reduce_mean(input_tensor, reduction_indices=none, keep_dims=false, name=none)
# input_tensor: the tensor need to reduce;
# reduction_indices = The dimensions to reduce. If None (the default), reduces all dimensions;
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
print("The error function: The cross-entropy cost function")
# Replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
print("The training algorithm: adam algorithm, learning rate is 0.0001")
# Calculate the accuracy
# tf.argmax:
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer())
for i in range(200):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        # eval means run
        # Using feed_dict to replace the placeholder tensors x, y_, and keep_prob with the training examples.
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("The accuracy of the %d training cycle: %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # when train_step run, it will apply ADAM optimizer to the parameters
    # Training the model can therefore be accomplished by repeatedly running train_step.

print("The accuracy of testing: %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))


