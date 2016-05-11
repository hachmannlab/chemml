from keras.models import Sequential
from keras.layers.core import Dense, Activation

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

def activations(activation,n):
    if isinstance(activation,list):
        if len(input_act_funcs) != n:
            raise Exception('List of activation function is of different \
                length from list of hidden_layers+1')
        else:
            return activation
    else:
        return [activation]*n

def train(data,target, hidden_layers, activtion, loss = 'mse', optimizer = 'rmsprop', 
            learning_rate = 0.001, training_epochs = 15, batch_size = 100, display_step = 1):
    N = data.shape[0]
    input_dim = data.shape[1]
    output_dim = target.shape[1]
    all_layers = [input_dim] + hidden_layers + [output_dim]
    activation_list = activations(activtion,len(all_layers)-1)
    
    # Create model
    model = Sequential()
    for prev_layer,layer,activ_func in zip(all_layers[:-1],all_layers[1:],activation_list):
        model.add(Dense(output_dim=layer, input_dim=prev_layer, init='uniform'))
        model.add(Activation(activ_func))
    
    model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'])
    model.fit(data,target, )   
    
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N = 2000
    X = (np.random.rand(N,1) - 0.5) * 10
    Y = np.sin(X)
    X_train = X[:N/2]
    X_test = X[N/2:]
    Y_train = Y[:N/2]
    Y_test = Y[N/2:]
    nnet = train(X_train,Y_train,[20,10],'tanh',learn_rate=0.001,n_epochs=50000,n_cores=2)
    print np.mean(np.abs(output(X_test,nnet) - Y_test),axis=0)
    plt.plot(X_test[:,0],output(X_test,nnet)[:,0],'x')
    plt.plot(X_test[:,0],Y_test[:,0],'o')
    plt.show()
