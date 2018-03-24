from functionDef import weight_variable, bias_variable, conv2d, max_pool_2x2
import tensorflow as tf
print("********************")
print("CNN Structure: ")
print("1 input layer, 2 convolution layers and 2 fully-connected layers")
print("********************")

# Placeholder: When we ask TensorFlow to run a computation, a value that we'll input.
# We wanna input any number of MNIST images, each flattened into a 784-dimensional vector
# 784: the dimensionality of a single flattened 28 by 28 pixel MNIST image
# We represent this as a 2-D tensor of floating-point numbers
# None indicates that the first dimension, corresponding to the batch size, can be of any size.
x = tf.placeholder(tf.float32, [None, 784])
# Define the input image: [batch_size, in_height, in_width, in_channels]
# Reshape x to a 4d tensor with the second and third dimensions corresponding to image width and height,
# and the final dimension corresponding to the number of color channels.
# [batch_size]: Size of the subset of examples to use when performing gradient descent during training.
# -1 means it will automatically compute based on the input
x_image = tf.reshape(x, [-1, 28, 28, 1])
# y_ is the expect output
# The expect output classes y_ will also consist of a 2d tensor,
# where each row is a one-hot 10-dimensional vector indicating which
# digit class (zero through nine) the corresponding MNIST image belongs to.
y_ = tf.placeholder(tf.float32, [None, 10])
print("Input image: ")
print("28*28, 1 channel")
print("********************")


###### The first convolutional layer######
# kernel: shape=[height, width, in_channels, out_channels]
# Out_channels = the no. of kernels
# There should be 32 kernels with 5x5 size
W_conv1 = weight_variable([5, 5, 1, 32])
# bias: Each output channel should have a bias
# Because there are 32 kernels -> 32 outputs -> 32 bias
b_conv1 = bias_variable([32])
# activation function: ReLu
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)  # Generate 32 of 28*28 output images
# pooling
h_pool1 = max_pool_2x2(h_conv1)
print("The first convolution layer: ")
print("kernel: 32 of 5*5; activation function: elu; pooling: mx_pool_2x2")
print("********************")

###### The Second convolutional layer######
# kernel: 32 input from last layer, 64 output -> 64 kernels of 5*5
W_conv2 = weight_variable([5, 5, 32, 64])
# bias
b_conv2 = bias_variable([64])
# activation function : ReLu
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
# pooling
h_pool2 = max_pool_2x2(h_conv2)
print("The second convolution layer: ")
print("kernel: 64 of 5*5; activation function: elu; pooling: mx_pool_2x2")
print("********************")

###### The Third Fully-Connected layer######
# kernel: input is 7*7*64, output is 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
# bias
b_fc1 = bias_variable([1024])
# flatten the input image into 1-D vector of 7*7*64
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# tf.matmul -> one-one multiply for two vector
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# To reduce over-fitting, we will apply dropout before the readout layer
# We create a placeholder for the probability that a neuron's output is kept during dropout.
# This allows us to turn dropout on during training, and turn it off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print("The third fully-connected layer: ")
print("activation function: elu; dropout: 0.5")
print("********************")

###### The Fourth Fully-Connected layer######
# kernel: input is 1024, output is 10 (10 labels/classes for output)
W_fc2 = weight_variable([1024, 10])
# bias: b is a 10-dimensional vector (because we have 10 classes)
b_fc2 = bias_variable([10])
# Use the Softmax as activation function
# y_conv is the actual output
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print("The fourth fully-connected layer: ")
print("Softmax Function")
print("********************")