# requires data
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.datasets.mnist import load_data

mnist = load_data()
print(mnist)

# check the type of dataset
print(type(mnist))

# array of training images
(x_train, y_train), (x_test, y_test) = load_data()
print('x_train: \n', x_train)
print('y_train: \n', y_train)
print('x_test: \n', x_test)
print('y_test: \n', y_test)

# data needs to be scaled in 2D but it's currently in 3D
# scaler = StandardScaler()
# for nums in x_train:
#     scaler.fit(nums)
# for nums in x_test:
#     scaler.fit(nums)

# number of images for training, testing and validation
print(f'x_train.shape: {x_train.shape}\nNumber of x_train samples: {x_train.shape[0]}')
print(f'y_train.shape: {y_train.shape}\nNumber of y_train samples: {y_train.shape[0]}')
print(f'x_test.shape: {x_test.shape}\nNumber of x_test samples: {x_test.shape[0]}')
print(f'y_test.shape: {y_test.shape}\nNumber of y_test samples: {y_test.shape[0]}')

# visualising the data
print('x_train[1] printing start...')
print(x_train[1])
print('x_train[1] printing end')
print('x_train printing start...')
print(x_train)
print('x_train printing end')

# view the first digit[1] in the dataset
plt.imshow(x_train[1].reshape(28, 28))
plt.show()
# view the grayscale of the digit
plt.imshow(x_train[1].reshape(28, 28), cmap='Greys')
plt.show()

# maximum and minimum value of the pixels in the image
print(x_train.max())
print(x_train.min())

# create the model
x = tf.compat.v1.placeholder(tf.float32,
                             shape=[None, 784])  # AttributeError: module 'tensorflow' has no attribute 'placeholder'
# 10 because 0-9 possible numbers
W = tf.Variable(tf.zeros[784, 10])
b = tf.Variable(tf.zeros[10])

# create the graph
y = tf.matmul(x, W) + b

# loss and optimizer
y_true = tf.compat.v1.placeholder(tf.float32, [None, 10])

# Cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# create the session ---> This is used in tensorflow 1.0, in 2.0 it's just model.fit(..,...,..,.,...)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # train the model for 1000 steps on the training set using built in batch feeder from mnist
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # evaluate the trained model on test data
    matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))
