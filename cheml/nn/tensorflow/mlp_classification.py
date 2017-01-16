import tensorflow as tf
from ...utils import choice, chunk
from ...utils import check_input

class mlp_classification(object):
    def __init__(self,nneurons, act_funcs, cost='mse', optimizer='GradientDescentOptimizer', validation_size=0.2,
                 learning_rate=0.001, training_epochs=1000, batch_size = 100, display_step = 10 ):
        """
        multilayer perceptron with tensorflow

        :param nneurons: list of integers
            describing how many neurons there are in each layer
        :param act_funcs: list of activation functions or just one function (string)
            should be same length as nneurons if a list
        :param cost: string (default = 'mse')
            List of available cost functions:
                - 'mse': mean squared error
                - 'scel': softmax cross_entropy with logits
        :param optimizer: string (default = 'GradientDescentOptimizer')
            List of available optimizers:
                - GradientDescentOptimizer
                - AdadeltaOptimizer
                - AdagradOptimizer
                - AdagradDAOptimizer
                - MomentumOptimizer
                - AdamOptimizer
                - FtrlOptimizer
                - ProximalGradientDescentOptimizer
                - ProximalAdagradOptimizer
                - RMSPropOptimizer
        :param validation_size: float between zero and one, optional (default = 0.2)
            size of data to be selected randomly for validation
        :param learning_rate:
        :param training_epochs:
        :param batch_size:
        :param display_step:
        """
        self.nneurons = nneurons
        self.act_funcs = act_funcs
        self.cost = cost
        self.optimizer = optimizer
        self.validation_size = validation_size
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step

    def train(self, X, Y):
        """
        make tensorflow variables and fit all the parameters. Then train the network parameters.

        :param X: pandas dataframe or numpy array
            input training data
        :param Y: pandas dataframe or numpy array
            output training data
        :return training accuracy: float
            validation accuracy: float
        """
        X_train, _ = check_input(X, 'Training input', format_out='ar')
        Y_train, _ = check_input(Y, 'Training output', n0=X_train.shape[0], format_out='ar')
        X_train, X_test, Y_train, Y_test = choice(X_train, Y_train, n=self.validation_size)
        X_test, _ = check_input(X_test, 'Testing input', n1=X_train.shape[1], format_out='ar')
        Y_test, _ = check_input(Y_test, 'Testing output', n0=X_test.shape[0], n1=Y_train.shape[1], format_out='ar')
        self.n_features = X_train.shape[1]
        self.n_outputs = Y_train.shape[1]
        N = X_train.shape[0]

        # tf Graph input
        x = tf.placeholder("float", [None, self.n_features])
        y = tf.placeholder("float", [None, self.n_outputs])
        self._initialize_weights()
        self.act_funcs = self._act_funcs_from_string()
        y_pred = self._predict(x)
        cost = self._loss_function(y_pred,y)
        optimizer = self._optimizer(cost)
        init = tf.initialize_all_variables()
        # Training cycle
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = N // self.batch_size
                # Loop over all batches
                it = chunk(range(len(X_train)),total_batch,X_train,Y_train)
                for i in xrange(total_batch):
                    batch_x, batch_y = it.next()
                    # Run optimization op (backprop) and cost op (to get loss value)
                    opt, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                          "{:.9f}".format(avg_cost))

        print self.weights
        print self.biases
        training_accuracy = self.score(X_train, Y_train, task = 'classification')
        validation_accuracy = self.score(X_test, Y_test, task = 'classification')
        print "Accuracy: ", "trainig: %f"%training_accuracy, "validation: %f"%validation_accuracy
        return training_accuracy, validation_accuracy

    def predict(self,X):
        """
        predict labels for the input data
        :param X: 2d array-like
            input data
        :return: predictions
            predicted labels
        """
        x = tf.placeholder("float", [None, self.n_features])
        y_pred = self._predict(x)
        with tf.Session() as sess:
            y_pred = sess.run(y_pred)
        return y_pred

    def score(self, X, Y, task='regression'):
        """
        returns mean absolute error if task is regression.
        returns fraction of correctly classified samples in case of classificaton task.

        :param X: 2d array-like
            input data with same number of features as the input training data
        :param Y: 1d/2d array-like
            labels with same dimension as input training labels
        :param task: string, default('regression')
            available options: 'regression' or 'classification'
        :return: score: 1d array-like
            the prediction accuracy of model using X as input data, compared to Y.
        """
        x = tf.placeholder("float", [None, self.n_features])
        y = tf.placeholder("float", [None, self.n_outputs])
        y_pred = self._predict(x)
        init = tf.initialize_all_variables()

        if task=='regression':
            accuracy = tf.reduce_mean(tf.abs(tf.subtract(y,y_pred)))
        elif task=='classification':
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        with tf.Session() as sess:
            sess.run(init)
            score = sess.run(accuracy, feed_dict={x: X, y: Y})
        return score

    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        complete_nneurons = [self.n_features] + self.nneurons + [self.n_outputs]
        for prev_nneurons, nneurons in zip(complete_nneurons[:-1], complete_nneurons[1:]):
            self.weights.append(tf.Variable(tf.random_normal([prev_nneurons, nneurons])))
            self.biases.append(tf.Variable(tf.random_normal([nneurons])))

    def _act_funcs_from_string(self):
        """
        translate activation functions input from user
        """
        act_func_dict = {'relu': tf.nn.relu,
                         'relu6': tf.nn.relu6,
                         # 'crelu': tf.nn.crelu,
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

    def _predict(self,x):
        hi = x
        for i in range(len(self.weights)-1):
            hi = self.act_funcs[i](tf.add(tf.matmul(hi, self.weights[i]), self.biases[i]))
        out_layer = tf.add(tf.matmul(hi, self.weights[-1]), self.biases[-1])
        return out_layer

    def _loss_function(self,y_pred,y):
        if self.cost == 'scel':
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
        elif self.cost == 'mse':
            cost = tf.reduce_mean(tf.square(tf.sub(y_pred, y)))
        return cost

    def _optimizer(self,cost):
        if self.optimizer == 'GradientDescentOptimizer':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'AdadeltaOptimizer':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'AdagradOptimizer':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'AdagradDAOptimizer':
            optimizer = tf.train.AdagradDAOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'MomentumOptimizer':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'FtrlOptimizer':
            optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'ProximalGradientDescentOptimizer':
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'ProximalAdagradOptimizer':
            optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=self.learning_rate).minimize(cost)
        elif self.optimizer == 'RMSPropOptimizer':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return optimizer

    def test(self):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        X=mnist.train.images
        Y=mnist.train.labels
        ta, va = self.train(X,Y)
        y_pred = self.predict(mnist.test.images)
        X_test = mnist.test.images
        Y_test = mnist.test.labels
        score = self.score(X_test, Y_test,task='classification')
        return score