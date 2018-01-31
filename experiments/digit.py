from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import numpy as np
import os, sys
import random
from PIL import Image

# set up all the tensorflow variables

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# alternative?
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

saver = tf.train.Saver()

# train the model
if sys.argv[1] == "train":
    print("running")

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    saver.save(sess, '/tmp/digit_model.ckpt')

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("the model accuracy is:")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# test the trained model with known array
if sys.argv[1] == "index":
    saver.restore(sess, "/tmp/digit_model.ckpt")

    if sys.argv[2] == "random":
        num = random.randrange(0, 500)
    else:
        num = int(sys.argv[2])
    image = mnist.test.images[num]
    data = np.array(image).reshape(1, 784)

    if len(sys.argv) > 3 and sys.argv[3] == "show":
        print("random array is:")
        print(image)

    print("index is:", num)
    guess = tf.argmax(y,1)
    confidence = tf.nn.softmax(tf.matmul(x, W) + b)
    print("guess is:",guess.eval(feed_dict={x: data})[0])
    i, = np.where( mnist.test.labels[num]==1 )
    print("actual is:", i[0])
    print("confidence is:", confidence.eval(feed_dict={x: data})[0])

# turn an image into an array
if sys.argv[1] == "file":
    saver.restore(sess, "/tmp/digit_model.ckpt")
    size = 28, 28

    directory = os.path.dirname(os.path.abspath(__file__))
    infile = os.path.join(directory, 'images/img{0}.jpg'.format(sys.argv[2]))
    outfile = os.path.join(directory, 'images/resize.jpg')

    image = Image.open(infile).convert('L')
    image.thumbnail(size, Image.ANTIALIAS)
    image.save(outfile, "JPEG")
    array = np.array(image, dtype=np.float)

    average = np.mean(array)
    def format(pixle):
        return ((pixle * -1) + 255) / 255 if pixle < (average / 2) else 0
    format = np.vectorize(format, otypes=[np.float])
    formatted_array = format(array)

    if len(sys.argv) > 3 and sys.argv[3] == "show":
        print("formatted array is:")
        print(formatted_array)

    guess = tf.argmax(y,1)
    confidence = tf.nn.softmax(tf.matmul(x, W) + b)
    print("guess is:",guess.eval(feed_dict={x: formatted_array.reshape(1, 784)})[0])
    print("actual is:", sys.argv[2])
    print("confidence is:", confidence.eval(feed_dict={x: formatted_array.reshape(1, 784)})[0])

# turn an image into an array
if sys.argv[1] == "loop":
    saver.restore(sess, "/tmp/digit_model.ckpt")
    size = 28, 28

    for i in range(0, 10):
        directory = os.path.dirname(os.path.abspath(__file__))
        infile = os.path.join(directory, 'images/img{0}.jpg'.format(i))
        outfile = os.path.join(directory, 'images/resize.jpg')

        image = Image.open(infile).convert('L')
        image.thumbnail(size, Image.ANTIALIAS)
        image.save(outfile, "JPEG")
        array = np.array(image, dtype=np.float)

        average = np.mean(array)
        def format(pixle):
            return ((pixle * -1) + 255) / 255 if pixle < (average / 2) else 0
        format = np.vectorize(format, otypes=[np.float])
        formatted_array = format(array)

        if len(sys.argv) > 3 and sys.argv[3] == "show":
            print("formatted array is:")
            print(formatted_array)

        guess = tf.argmax(y,1)
        confidence = tf.nn.softmax(tf.matmul(x, W) + b)
        print("guess is:",guess.eval(feed_dict={x: formatted_array.reshape(1, 784)})[0])
        print("actual is:", i)
        # print("confidence is:", confidence.eval(feed_dict={x: formatted_array.reshape(1, 784)})[0])

if sys.argv[1] == "noise":
    saver.restore(sess, "/tmp/digit_model.ckpt")
    saver.restore(sess, "/tmp/digit_model.ckpt")

    random_array = np.random.rand(1,784)

    if len(sys.argv) > 2 and sys.argv[2] == "show":
        print("random array is:")
        print(random_array)

    guess = tf.argmax(y,1)
    confidence = tf.nn.softmax(tf.matmul(x, W) + b)
    print("guess is:",guess.eval(feed_dict={x: random_array})[0])
    print("actual is:", "NA")
    print("confidence is:", confidence.eval(feed_dict={x: random_array})[0])

if sys.argv[1] == "real":
    saver.restore(sess, "/tmp/digit_model.ckpt")

    # index 85

    real_array = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.01176471,0.03921569,0,0,0,0,0,0,0,0,0,0.2627451,0.9960785,0.83921576,0.09803922,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2392157,0.89019614,0.13725491,0,0,0,0,0,0,0,0.01960784,0.7372549,0.9921569,0.9921569,0.64705884,0.01176471,0,0,0,0,0,0,0,0,0,0,0,0.02352941,0.7137255,0.9921569,0.7058824,0.11764707,0,0,0,0,0,0,0.19215688,0.9921569,0.9921569,0.9921569,0.9921569,0.09411766,0,0,0,0,0,0,0,0,0,0,0,0.14509805,0.9921569,0.9921569,0.9921569,0.5019608,0,0,0,0,0,0,0.28235295,0.9921569,0.9921569,0.9921569,0.7372549,0.03137255,0,0,0,0,0,0,0,0,0,0,0,0.14509805,0.9921569,0.9921569,0.9921569,0.65882355,0,0,0,0,0,0,0.6627451,0.9921569,0.9921569,0.91372555,0.24313727,0,0,0,0,0,0,0,0,0,0,0,0,0.28235295,0.9921569,0.9921569,0.9921569,0.65882355,0,0,0,0,0,0.04313726,0.7607844,0.9921569,0.9921569,0.65882355,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6156863,0.9921569,0.9921569,0.9921569,0.65882355,0,0,0.16078432,0.20000002,0.53333336,0.654902,0.9921569,0.9921569,0.9921569,0.65882355,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6156863,0.9921569,0.9921569,0.9921569,0.78823537,0.56078434,0.627451,0.9215687,0.9333334,0.9921569,0.9921569,0.9921569,0.9921569,0.9921569,0.48235297,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6156863,0.9921569,0.9921569,0.9921569,0.9921569,0.9921569,0.9921569,0.9921569,0.9960785,0.9921569,0.9921569,0.9921569,0.9921569,0.73333335,0.01960784,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6156863,0.9921569,0.9921569,0.9921569,0.9921569,0.9921569,0.9921569,0.9921569,0.9960785,0.9921569,0.9921569,0.9921569,0.9921569,0.7058824,0,0,0,0,0,0,0,0,0,0,0,0,0,0.09411766,0.9803922,0.9960785,0.9960785,0.9960785,0.9960785,0.9960785,0.9960785,0.9960785,0.97647065,0.97647065,0.9960785,0,0.9960785,0.46274513,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.33333334,0.9176471,0.9921569,0.9921569,0.9921569,0.9921569,0.9058824,0.67058825,0.20392159,0.5686275,0.9921569,0.9921569,0.9921569,0.28235295,0.07843138,0.01568628,0,0,0,0,0,0,0,0,0,0,0,0,0,0.07450981,0.7176471,0.85098046,0.6901961,0.37647063,0.03529412,0,0,0.2784314,0.9921569,0.9921569,0.9921569,0.8862746,0.8745099,0.12156864,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.09803922,0.9921569,0.9921569,0.9921569,0.9921569,0.5686275,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.43921572,0.9921569,0.9921569,0.9921569,0.6745098,0.05490196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.03921569,0.8941177,0.9921569,0.9921569,0.92549026,0.16862746,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.20784315,0.9921569,0.9921569,0.9921569,0.59607846,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.34509805,0.9921569,0.9921569,0.82745105,0.03921569,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0509804,0.9921569,0.9921569,0.80392164,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.02745098,0.79215693,0.9921569,0.5764706,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    real_array = np.array(real_array, dtype=np.float)

    if len(sys.argv) > 2 and sys.argv[2] == "show":
        print("array is:")
        print(real_array)

    guess = tf.argmax(y,1)
    confidence = tf.nn.softmax(tf.matmul(x, W) + b)
    print("guess is:",guess.eval(feed_dict={x: real_array})[0])
    print("actual is:", 4)
    print("confidence is:", confidence.eval(feed_dict={x: real_array})[0])
