import numpy as np
import tensorflow as tf
import random

## https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/multilayer_perceptron.py

#input data
N = 2000
X = (np.random.rand(N,1) - 0.5) * 10
Y = np.sin(X)
X_train = X[:N/2]
X_test = X[N/2:]
Y_train = Y[:N/2]
Y_test = Y[N/2:]
def chunk(xs, n):
    """
    The use case: chunk_list= list( chunk(range(10),3) )
    """
    ys = list(xs)
    random.shuffle(ys)
    size = len(ys) // n
    leftovers= ys[size*n:]
    for c in xrange(n):
        if leftovers:
           extra= [ leftovers.pop() ] 
        else:
           extra= []
        yield ys[c*size:(c+1)*size] + extra
    

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 4 # 1st layer num features
n_hidden_2 = 2 # 2nd layer num features
n_input = 1 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
    return tf.matmul(layer_2, _weights['out']) + _biases['out']
    
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        mini_batches = list (chunk(range(len(X_train)), batch_size))
        # Loop over all batches
        for i,mini_batch in enumerate(mini_batches):
            batch_xs, batch_ys = X_train[mini_batch],Y_train[mini_batch]
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/len(mini_batches)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "learning_rate=", "{:.4f}".format(learning_rate)
            
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: X_test, y: Y_test})