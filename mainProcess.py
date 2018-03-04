# y = softmax(Wx+b)
# Define W, x, and b
# x is the image, y is the "labels"
# Each image in MNIST has a corresponding label, a number between 0 and 9 representing the digit drawn in the image.
# Hence, there would be only 10 possible labels(classes) for a given image.
# For this implementation, we want our labels as "one-hot vectors".
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.
# In this case, the nth digit will be represented as a vector which is 1 in the nth dimension.
# 3 = [0 0 0 1 0 0 0 0 0 0]

from CNNLayers import x, y_, keep_prob
from lossFunction import accuracy, train_step
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
print("Data Directory:" + FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True) # MNIST is returned by read_data_sets
# The images are returned as a 2-D NumPy array of size [batch_size, 784] (since there are 784 pixels in an MNIST image),
# and the labels are returned as 2-D NumPy array of size [batch_size, 10]
# (if read_data_sets() was called with one_hot=True).
# Otherwise, the labels are returned as a 1-D NumPy array of size [batch_size]

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
for i in range(3000):
    batch = mnist.train.next_batch(50) # Pick 50 datasets for everytime train
    # mnist.train.next_batch(batch_size)
    # Returns a tuple of 2 arrays, where the first represents a batch of batch_size MNIST images,
    # and the second represents a batch of batch-size labels corresponding to those images.
    if i % 100 == 0:
        # eval means run
        # Using feed_dict to replace the placeholder tensors x, y_, and keep_prob.
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("The accuracy of the %d training cycle: %g" % (i, train_accuracy))

    # when train_step run, it will apply ADAM optimizer to the parameters
    # Training the model can therefore be accomplished by repeatedly running train_step.
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("The accuracy of the testing: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
