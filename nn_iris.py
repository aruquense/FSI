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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

train_set_x=x_data[:105]
train_set_y=y_data[:105]
valid_set_x=x_data[106:128]
valid_set_y=y_data[106:128]
test_set_x=x_data[129:]
test_set_Y=y_data[129:]


print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

#4 entradas-5neuronas
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)
#5 entradas-3neuronas
W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
#h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#Media
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

epoch=0
batch_size = 10
#for epoch in xrange(100):
error=100
#while error>1 and epoch<500 :
diferencia=[999,error]
while (diferencia[0]-diferencia[1]) > 0.001 and epoch < 500:
    epoch+=1
    for jj in xrange(len(train_set_x) / batch_size):
        batch_xs = train_set_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_set_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: valid_set_x, y_: valid_set_y})
    print "Epoch #:", epoch, "Error: ", error

    if(diferencia[1]>error):
        diferencia[0]=diferencia[1]
        diferencia[1]=error

    print "----------------------------------------------------------------------------------"
fallo = sess.run(loss, feed_dict={x: test_set_x, y_: test_set_Y})
print "Test Error: ", fallo
print "Precision: ", sess.run(accuracy, feed_dict={x: test_set_x, y_: test_set_Y}),"%"
