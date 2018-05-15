import gzip
import pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.pyplot as plt

# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print(train_y[57])

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

# HIDDEN LAYERS
W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

# OUTPUT
W3 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b3 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h2 = tf.nn.sigmoid(tf.matmul(h, W2) + b2)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h2, W3) + b3)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")


def difference_in_error(errors):
    if len(errors) > 2:
        if abs(errors[-1] - errors[-2]) < 0.1 and errors[-1] < 0.05:
            return False
    return True


train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)

batch_size = 20
epochs = 0
train_errors = []
valid_errors = []

while epochs < 50 and difference_in_error(valid_errors):
    # Cada iteración entrena la red con 20 muestras (20 números)
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    train_error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}) / batch_size
    # train_error = sess.run(loss, feed_dict={x: train_x, y_: train_y}) / len(train_x)
    train_errors.append(train_error)
    validation_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y}) / len(valid_y)
    valid_errors.append(validation_error)

    print("Epoch #:", epochs, "Train error: ", train_error, "Validation error: ", validation_error)
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print(b, "-->", r)
    print("----------------------------------------------------------------------------------")
    epochs += 1

# NET ACCURACY
mistakes = 0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    for i in range(10):
        if b[i] != round(r[i]):
            mistakes += 1
            break

number_of_samples = len(test_x)
print("Hit rate:", ((number_of_samples - mistakes) / number_of_samples) * 100, "%")

# GRAPH
plt.plot(train_errors)
plt.plot(valid_errors)
plt.legend(["Train Error", "Validation Error"])
plt.show()
