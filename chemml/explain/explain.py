from multiprocessing.sharedctypes import Value
from turtle import back
from typing import Type
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
from .visualize import plot_lime, plot_lrp, plot_shap_local, plot_shap_global

class Explain(object):

    def __init__(self, X_instance, dnn_obj, feature_names):
        '''
        Parameters
        ----------
        X_instance: torch.Tensor or numpy.ndarray
            Instance to be explained by XAI method, can be single instance or multiple instances

        dnn_obj: torch.nn.Module
            Trained deep neural network

        feature_names: list
            list of feature names/column names/descriptor names
        '''

        if type(X_instance) == torch.Tensor:
            self.X_instance = X_instance
        elif type(X_instance) == np.ndarray:
            self.X_instance = torch.tensor((X_instance),dtype=torch.float)
        # elif X_instance.shape[1] != len(feature_names):
        #     raise ValueError('Shape of X_instance should either be (1,no. of features) or (m, no. of features)')
        else:
            raise TypeError('X_instance has to be a numpy array or pytorch tensor')
        

        # check of dnn_obj is of chemml model type
        if isinstance(dnn_obj,torch.nn.Module):
            self.dnn_obj = dnn_obj
        else:
            raise TypeError('dnn_obj has to be a pytorch deep neural network')

        if type(feature_names) != list:
            raise TypeError('feature_names should be a list of column names for X_instance')
        if self.X_instance.ndim > 1 and self.X_instance.shape[1] != len(feature_names):
            raise ValueError('feature_names should be of the same size as X_instance')
        elif self.X_instance.ndim == 1:
            if len(self.X_instance) != len(feature_names):
                raise ValueError('feature_names should be of the same size as X_instance')
            else:
                self.X_instance = self.X_instance.reshape(1,-1)
        self.feature_names = feature_names


    def LRP(self, strategy='zero', global_relevance=False):
        '''
        Parameters
        ----------
        strategy: str, default = 'zero'
            can be 'zero', 'eps' and 'composite'
            LRP strategy from paper (link)

        global_relevance: bool, default = False
            True if global aggregation is required

        Returns
        -------       
        rel_df: pandas dataframe
            Local relevance scores for individual X_instance

        global_scores: pandas dataframe or None
            global (mean) actual and absolute relevance scores aggregated for all X_instances
            None if global_relevance = False
        '''

        def _lrp_local(X):

            # Get the list of layers of the network
            layers = [module for module in self.dnn_obj.modules()][1:]
            # Propagate the input
            L = len(layers)
            A = [X] + [X] * L # Create a list to store the activation produced by each layer
            for layer in range(L):
                A[layer + 1] = layers[layer].forward(A[layer])
            # Get the relevance of the last layer using the highest classification score of the top layer
            T = A[-1].cpu().detach().numpy()

            # Changing confidence intervals to discrete values (i.e., masksing): relevant for \
            # classification problems
            # index = T.index(max(T))
            # T = np.abs(np.array(T)) * 0
            # T[index] = 1
            
            T = torch.FloatTensor(T)
            # Create the list of relevances with (L + 1) elements and assign the value of the last one 
            R = [None] * L + [(A[-1].cpu() * T).data + 1e-6]
            # Propagation procedure from the top-layer towards the lower layers
            for layer in range(0, L)[::-1]:
                if isinstance(layers[layer],torch.nn.Linear):
                    # Specifies the rho function that will be applied to the weights of the layer

                    if strategy == 'composite':
                        if layer == 0:
                            eps = 0
                            rho = lambda p:p + 0.5 * p.clamp(min=0,max=1) - 0.5 *p.clamp(min=-1,max=0)
                        else:
                            eps = 0.25
                            rho = lambda p:p
                    elif strategy == 'zero':
                        rho = lambda p: p
                        eps = 0
                    elif strategy == 'eps':
                        rho = lambda p: p
                        eps = 0.25
                    else:
                        raise ValueError("strategy should be 'zero', 'eps' or 'composite'")

                    # clamp: clamp all elements in input into range[min, max] 
                    # rho = lambda p: p + 0.25 * p.clamp(min=0)
                    # rho = lambda p: p + 0.25
                    

                    A[layer] = A[layer].data.requires_grad_(True)

                    # Step 1: Transform the weights of the layer and executes a forward pass
                    z = eps + self._newlayer(layers[layer], rho).forward(A[layer]) + 1e-9

                    # Step 2: Element-wise division between the relevance of the next layer and the denominator
                    s = (R[layer + 1].to(device) / z).data
                    
                    # Step 3: Calculate the gradient and multiply it by the activation layer
                    (z * s).sum().backward()
                    c = A[layer].grad  										   
                    R[layer] = (A[layer] * c).cpu().data  
                    
                else:
                    R[layer] = R[layer + 1]
            
            # Return the relevance of the input layer
            return R[0]

        self.explainer = 'lrp'
        device = 'cpu'
        X = self.X_instance
        scores = []
        for i in range(0,X.shape[0]):
            rel = _lrp_local(X[i]).numpy()
            scores.append(rel)
        rel_df = pd.DataFrame(scores,columns=self.feature_names)
        if global_relevance == True:
            m_scores = rel_df.mean().values
            m_abs_scores = rel_df.abs().mean().values
            global_scores = pd.DataFrame({'Mean Absolute Relevance Score':m_abs_scores,'Mean Relevance Score':m_scores},index=self.feature_names)
            global_scores = global_scores.sort_values(by='Mean Absolute Relevance Score',ascending=False)
            return rel_df, global_scores
        return rel_df, None


    def DeepSHAP(self,X_background):
        '''
        Parameters
        ----------
        X_background: numpy ndarray or torch.Tensor
            input features of the reference or baseline sample(s) from the data
    
        Returns
        -------       
        rel_df: pandas dataframe
            Local relevance scores for the X_instance

        shap_obj: shap.DeepExplainer object
            An instance of the DeepSHAP model required to obtain the expectation value for waterfall plots

        '''
        self.explainer = 'deepshap'

        if type(X_background) == np.ndarray:
            X_background = torch.tensor((X_background),dtype=torch.float)
        elif type(X_background) != torch.Tensor:
            raise TypeError('background has to be a numpy array or pytorch tensor')

        import shap

        shap_obj = shap.DeepExplainer(self.dnn_obj,X_background)
        shap_values = shap_obj.shap_values(self.X_instance)
        rel_df = pd.DataFrame(shap_values.reshape(len(self.X_instance),len(self.feature_names)),columns=self.feature_names)
        return rel_df, shap_obj


    def LIME(self, training_data):
        '''
        https://lime-ml.readthedocs.io/en/latest/lime.html
        https://pypi.org/project/lime/
        Parameters
        ----------
        training_data: numpy.ndarray, default = None
            feature values for training data, required only for LIME
        
        Returns
        -------
        scores_all: list
            list of pandas dataframes with relevance scores for each X_instance
        '''
        if training_data is None or type(training_data) != np.ndarray:
            raise TypeError('training data has to be an numpy ndarray')

        self.explainer = 'lime'
        import lime
        import lime.lime_tabular
        lime_obj = lime.lime_tabular.LimeTabularExplainer(training_data, feature_names=self.feature_names,
                                                            feature_selection='lasso_path',
                                                            class_names=['density'],
                                                            verbose=True,mode='regression',random_state=42)
        scores_all = []
        for i in range(len(self.X_instance)):
            rel = lime_obj.explain_instance(self.X_instance[i].numpy().reshape(-1,),self._predict_for_lime, num_features=len(self.feature_names))
            rel = rel.as_list()
            labels = []
            relevances = []
            for i in rel:
                labels.append(i[0])
                relevances.append(i[1])
            scores = pd.DataFrame({'labels':labels, 'local_relevance':relevances})
            scores_all.append(scores)

        return scores_all


    def _newlayer(self,layer, g, alpha_beta=False):
        """Clone a layer and pass its parameters through the function g."""
        if alpha_beta:
            pass
        else:
            layer = copy.deepcopy(layer)
            layer.weight = torch.nn.Parameter(g(layer.weight))
            layer.bias = torch.nn.Parameter(g(layer.bias))
        return layer


    def _predict_for_lime(self, X_test):
        X_test = torch.tensor((X_test),dtype=torch.float)
        y_hat = self.dnn_obj(X_test)
        y_hat = y_hat.detach().numpy()
        return y_hat


    def plot(self, local, rel_df, max_display=10, shap_obj=None):
        '''
        Plots local or global relevance scores

        Parameters
        ----------
        local: bool
            True if local relevance plots are required and false if global relevance plots are required
        
        rel_df: pandas dataframe
            global or local relevance scores; use un-modified relevance dataframes returned from LRP, DeepSHAP, or LIME methods
        
        max_display: int, default=10
            no. of most impactful features to show in the plot
        
        shap_obj: shap.DeepExplainer object
            An instance of the DeepSHAP model required to obtain the expectation value for waterfall plots
        Returns
        -------
        f: matplotlib.figure object or list of matplotlib.figure objects

        '''
        if not isinstance(rel_df, pd.core.frame.DataFrame):
            raise TypeError('Relevances should be in the form of a pandas dataframe as returned by LIME, DeepSHAP, and LRP methods')

        if self.explainer == 'lime':
            if local==False:
                raise ValueError('Global relevances cannot be generated using LIME scores')
            else:
                if list(rel_df.columns) != ['labels','local_relevance']:
                    raise ValueError('Relevance dataframe must have columns labels and local_relevance for lime')

                rel_df['abs'] = rel_df['local_relevance'].abs()
                rel_df.sort_values(by='abs',ascending=False,inplace=True)
                f = plot_lime(rel_df=rel_df, max_features=max_display)
        
        elif self.explainer == 'lrp':
            if local == True:
                if rel_df.shape[1] != len(self.feature_names):
                    raise ValueError('Relevance dataframe has been modified, the number of features are not the same as the input')
                rel_df_t = rel_df.transpose()
                f = []
                for i in range(rel_df.shape[0]):
                    rel_df_new = rel_df_t[rel_df_t.columns[i:i+1]]
                    rel_df_new['abs'] = rel_df_new[rel_df_new.columns[0]].abs()
                    rel_df_new = rel_df_new.sort_values(by='abs',ascending=False)
                    rel_df_new.drop(labels='abs',axis=1,inplace=True)

                    f.append(plot_lrp(rel_df_new, max_display))
            else:
                if list(rel_df.columns) != ['Mean Absolute Relevance Score', 'Mean Relevance Score']:
                    raise ValueError('Relevance dataframe must have columns Mean Absolute Relevance Score and Mean Relevance Score for LRP')
                
                rel_df.drop(labels='Mean Absolute Relevance Score',axis=1,inplace=True)
                f = plot_lrp(rel_df, max_display)

        elif self.explainer == 'deepshap':
            if isinstance(self.X_instance, torch.Tensor):
                features = self.X_instance.detach().numpy()
            if local == True:
                expected_value = shap_obj.expected_value[0]
                f = []
                for i in range(len(rel_df)):
                    f.append(plot_shap_local(expected_value=expected_value, shap_values=rel_df.loc[i,:], features=features[i], feature_names=self.feature_names, max_display=max_display))
            else:
                f = plot_shap_global(data=features,shap_values=rel_df.values,max_display=max_display,feature_names=self.feature_names)
            
        return f
