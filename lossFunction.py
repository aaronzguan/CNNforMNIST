import tensorflow as tf
from CNNLayers import y_, y_conv

# Define the error function / Loss function
# Loss indicates how bad the model's prediction was on a single example
# so we try to minimize the loss while training across all the examples
# Our loss function is the cross-entropy between the target and the softmax activation function applied to the model's prediction
# tf.reduce_mean(input_tensor, reduction_indices=none, keep_dims=false, name=none)
# [input_tensor]: the tensor need to reduce;
# [reduction_indices] = The dimensions to reduce. If None (the default), reduces all dimensions;
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
print("The error function: The cross-entropy cost function")
print("********************")
# Replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
print("The training algorithm: adam algorithm with 0.0001 learning rate")
print("********************")
# Calculate the accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))