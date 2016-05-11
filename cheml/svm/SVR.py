import numpy as np
import cvxopt
import sys
import copy

def linear_kernel(x1,x2): 
    return np.dot(x1,x2)

def polynomial_kernel(x1,x2,p): 
    return np.power(1 + np.sum(x1*x2),p)

def gaussian_kernel(x1,x2,sigma): 
    return np.exp(-np.linalg.norm(x1-x2)**2 / float(2 * (sigma ** 2)))

def laplacian_kernel(x1,x2,sigma): 
    return np.exp(-np.linalg.norm(x1-x2) / float(sigma))
                 
def check_array(X,name,n0=None,n1=None):
    """
    Makes sure that array is correct
    Inputs:
    X - array to be checked
    n0 - number of data entries
    n1 - number of features
    name - name of array (e.g. training input)
    Output:
    array converted to floats
    """
    if not (isinstance(X, np.ndarray) and X.ndim == 2):
        raise Exception(name+' needs to be two dimensional numpy array')
    elif not np.can_cast(X.dtype,np.float,casting='same_kind'):
        raise Exception(name+' cannot be cast to floats')
    if n0 and X.shape[0] != n0:
        raise Exception(name+' has an invalid number of data entries')
    if n1 and X.shape[1] != n1:
        raise Exception(name+' has an invalid number of feature entries')
    return X.astype(float)


class svr(object):
    def __init__(self, kernel='linear', C=1, epsilon=0.1, p=3, sigma=0.5):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.p = p
        self.sigma = sigma

               
    def fit(self, X, y):
        X = check_array(X,'X')
        y = check_array(Y_train,'y',n0=X.shape[0])
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
                
        if self.kernel == 'linear':
            self.Kernel_Func = linear_kernel
        elif self.kernel == 'polynomial':
            self.Kernel_Func = lambda x1,x2,p=self.p: polynomial_kernel(x1,x2,p)
        elif self.kernel == 'gaussian': 
            self.Kernel_Func = lambda x1,x2,sigma=self.sigma: gaussian_kernel(x1,x2,sigma)
        elif self.kernel == 'laplacian':
            self.Kernel_Func = lambda x1,x2,sigma=self.sigma: laplacian_kernel(x1,x2,sigma)
              
        H = np.zeros((self.n_samples,self.n_samples))
        for i in range(self.n_samples):
            for j in range(i,self.n_samples):
                H[i,j] = self.Kernel_Func(X[i],X[j])
                H[j,i] = H[i,j]
        
        HH1 = np.concatenate((H, -H), axis=1)
        HH2 = np.concatenate((-H, H), axis=1)
        HH = cvxopt.matrix(np.concatenate((HH1, HH2), axis=0))
        f = cvxopt.matrix(np.concatenate((-y, y), axis=0) + self.epsilon)
        A = np.concatenate((np.ones((1,self.n_samples)), -np.ones((1,self.n_samples))), axis=1)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.concatenate((np.diag(np.ones(2 * self.n_samples)) * -1, np.diag(np.ones(2 * self.n_samples))), axis=0))
        h = cvxopt.matrix(np.concatenate((np.zeros(2 * self.n_samples), np.ones(2 * self.n_samples) * self.C), axis=0))
        
        solution = cvxopt.solvers.qp(HH, f, G, h, A, b)
        self.alpha = np.reshape(solution['x'], (2 * self.n_samples , 1))
        almost_zero = abs(self.alpha)<max(abs(self.alpha))*10**-5
        self.alpha[almost_zero] = 0.0
        alpha_plus = self.alpha[:self.n_samples]
        alpha_minus = self.alpha[self.n_samples:]
        self.eta = alpha_plus - alpha_minus
        self.S = np.where((alpha_plus + alpha_minus > 0) & (alpha_plus + alpha_minus < self.C))
        self.XS = X[self.S[0]]
        
        #w: weights for linear kernel
        if self.kernel == 'linear':
            for i in range(self.n_samples):
                self.w = np.zeros(self.n_features)
                for i in self.S[0]:
                    self.w += self.eta[i]*X[i] 
        else:
            self.w = None
                
        #b: intercept
        self.b = 0
        if self.kernel == 'linear':
            for i in self.S[0]:
                self.b += y[i][0] - sum(self.w * X[i]) - np.sign(self.eta[i])[0]*self.epsilon
        else:
            y_predict = np.zeros(self.n_samples)
            for j in range(self.n_samples):
                y_predict[j] = self._predict_singlepoint(X[j])
            for i in self.S[0]:
                self.b += y[i][0] - y_predict[i] - np.sign(self.eta[i])[0]*self.epsilon
        self.b /= float(len(self.S[0]))
        
        
    def _predict_singlepoint(self, x):
        eta = self.eta[self.S[0]]
        y = 0
        for i in range(len(eta)):
            y += eta[i]*self.Kernel_Func(x,self.XS[i])
        return y[0]
    
    
    def predict(self, X):
        """
        Gives the output of the trained model. 
        """
        if self.kernel == 'linear':
            return np.dot(self.w,X.T) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                y_predict[i] = self._predict_singlepoint(X[i])
            return y_predict + self.b
        
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt	
    N = 200
    X = (np.random.rand(N,1) - 0.5) * 10
    Y = np.sin(X) + 0.2 * (np.random.random((N,1))-0.5)
    X_train = X[:N/2]
    X_test = X[N/2:]
    Y_train = Y[:N/2]
    Y_test = Y[N/2:]
    model = svr('gaussian', C=1, epsilon=0.2, p=3, sigma=0.3)
    model.fit(X_train,Y_train)
    print np.mean(np.abs(model.predict(X_test) - Y_test),axis=0)
    plt.plot(X_test[:,0],model.predict(X_test),'x')
    plt.plot(X_test[:,0],Y_test[:,0],'o')
    plt.savefig('plot.png')
#     plt.show()

