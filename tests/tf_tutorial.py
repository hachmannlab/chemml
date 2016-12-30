from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist.train : 55000 * 784
# mnist.test : 10000 * 784
# mnist.validation : 5000 * 784

print mnist.train.ys
