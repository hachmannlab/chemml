# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
from ...utils import choice
from ...utils import check_input


class tf_mlp(object):
    def __init__(self,nneurons, act_funcs, validation_size=0.2, learning_rate=0.001, training_epochs=25, batch_size = 100, display_step = 10 ):
        """
        multilayer perceptron with tensorflow

        :param nneurons: list of integers
            describing how many neurons there are in each layer
        :param act_funcs: list of activation functions or just one function (string)
            should be same length as nneurons if a list
        :param validation_size: float between zero and one, optional (default = 0.2)
            size of data to be selected randomly for validation
        :param learning_rate:
        :param training_epochs:
        :param batch_size:
        :param display_step:
        """
        self.nneurons = nneurons
        self.act_funcs = act_funcs
        self.validation_size = validation_size
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        # n_hidden_1 = 256 # 1st layer number of features
        # n_hidden_2 = 256 # 2nd layer number of features
        # n_input = 784 # MNIST data input (img shape: 28*28)
        # n_classes = 10 # MNIST total classes (0-9 digits)

    def fit(self, X, Y):
        """

        :param X: pandas dataframe or numpy array
            input training data
        :param Y: pandas dataframe or numpy array
            output training data
        """
        X_train, _ = check_input(X, 'Training input', format_out='ar')
        Y_train, _ = check_input(Y, 'Training output', n0=X_train.shape[0], format_out='ar')
        X_train, X_test, Y_train, Y_test = choice(X_train, Y_train, n=validation_size)
        X_test, _ = check_input(X_test, 'Testing input', n1=X_train.shape[1], format_out='ar')
        Y_test, _ = check_input(Y_test, 'Testing output', n0=X_test.shape[0], n1=Y_train.shape[1], format_out='ar')
        self.n_features = X_train.shape[1]
        self.n_outputs = Y_train.shape[1]
        N = X_train.shape[0]

        # tf Graph input
        x = tf.placeholder("float", [N, n_features])
        y = tf.placeholder("float", [N, n_outputs])
        self._initialize_weights()
        self._act_funcs_from_string()
        self._create_network()

    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        complete_neurons = [self.n_features] + self.nneurons + [self.n_outputs]
        for prev_nneurons, nneurons in zip(complete_nneurons[:-1], complete_nneurons[1:]):
            self.weights.append(tf.Variable(tf.random_normal([prev_nneurons, nneurons])))
            self.biases.append(tf.Variable(tf.random_normal([nneurons])))

    def _act_funcs_from_string(self):
        """
        translate activation functions input from user
        """
        act_func_dict = {'relu': tf.nn.relu,
                         'relu6': tf.nn.relu6,
                         'crelu': tf.nn.crelu,
                         'elu': tf.nn.elu,
                         'softplus': tf.nn.softplus,
                         'softsign': tf.nn.softsign,
                         'sigmoid': tf.sigmoid,
                         'tanh': tf.tanh}
        if isinstance(self.act_funcs, list):
            if len(self.act_funcs) != len(self.weights)-1:
                msg = 'List of activation function is of different length from list of neuron numbers'
                raise Exception(msg)
            else:
                return map(lambda x: act_func_dict[x], self.act_funcs)
        else:
            return [act_func_dict[self.act_funcs]] * (len(self.weights) - 1)

    def _create_network(self):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

