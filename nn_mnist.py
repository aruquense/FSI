import gzip
import cPickle

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
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

#etiquetas
train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 28*28])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

#28*28 entradas-10neuronas
W1 = tf.Variable(np.float32(np.random.rand(28*28, 10)) * .1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * .1)

#10entradas-10neuronas una por cada numero

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * .1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * .1)




#Funciones de activacion
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)


#funcion de perdida Sumatorio del cuadrado de las diferencias
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
batch_size = 20
#for epoch in xrange(100):
error=10000
diferencia=[99999,error]
error_list = []
while (diferencia[0]-diferencia[1]) > 1 and epoch < 50:
    epoch+=1
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    error_list.append(error)
    print "Epoch #:", epoch, "Error: ", error

    # Como es una condicion del bucle, para que no se me salga
    # Si la diferencia empieza a subir sale por numero de epocas
    if(diferencia[1]>error):
        diferencia[0]=diferencia[1]
        diferencia[1]=error

    print "----------------------------------------------------------------------------------"
fallo = sess.run(loss, feed_dict={x: test_x, y_: test_y})
print "Test Error: ", fallo
print "Precision: ", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}),"%"







# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#imprimir funcion de perdida matplotlib


#plt.imshow(train_x[11].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[11]


# TODO: the neural net!!

plt.ylabel('Error')
plt.xlabel('Epoca')
plt.plot(error_list)
plt.savefig('Grafica.png')
#plt.show()