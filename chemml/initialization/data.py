from builtins import range
import pandas as pd
import numpy as np

def _group_parser(indices, pop):
    # indices is a dictionary
    sel_ind = []
    for i in indices:
        if len(indices[i]) <= pop :
            sel_ind += indices[i]
        else:
            sel = list(np.random.choice(indices[i],size=pop,replace=False))
            sel_ind += sel
    return sel_ind

class Trimmer(object):
    """ cut unnecessary parts of the data
    
    Parameters
    ----------
    type: string, optional (default="margins")
        
        list of types:
        - random: remove part of data randomly
        - margins: cut from both top and bottom of data set
        - top: cut only top margin
        - bottom: cut only bottom margin
    
    sort: boolean, optional (default=True)
        If True data will be sorted by target values before trimming.
    
    cut: float in (0,1) range, optional (default=0.05)
        fraction of data set size to be trimmed.
    
    shuffle: Boolean, optional (default=True)
        To shuffle the data before sampling. Effective only if sort is False. 
    
    Attributes
    ----------
    Ncut: int
        number of removed data points
    
    selected_indices_: list
        Axis labels of data points that have been drawn randomly. Available only
        if type is 'random'.
          
    Returns
    -------
    data and target
    """
    def __init__(self, type="margins", sort = True,
                 cut = 0.05, shuffle = True):
        self.type = type
        self.sort = sort
        self.cut = cut
        self.shuffle = shuffle
        
    def fit_transform(self, data, target):
        """
        Fit the trimmer on df.
        """
        df = pd.concat([data, target], axis=1)
        if target.columns[0] in data.columns:
            cols = list(df.columns)
            col = 'target'
            while col in cols:
                col += 't'
            cols[-1] = col
            df.columns = cols
        else:
            col = target.columns[0]
        
        if self.sort == True:
            df.sort_values(col,axis=0,inplace=True)
            df.index = pd.Index(range(len(df)))
            data = df.iloc[:,:-1]
            target = pd.DataFrame(df.iloc[:,-1])
            
        elif self.shuffle == True:
            df = df.reindex(np.random.permutation(df.index))
            df.index = pd.Index(range(len(df)))            
            data = df.iloc[:,:-1]
            target = pd.DataFrame(df.iloc[:,-1])
                
        Nsamples = len(data)
        self.Ncut_ = int(self.cut * Nsamples) 
        if self.type == 'random':
            self.selected_indices_ = np.random.choice(range(0, Nsamples), Nsamples-self.Ncut_,replace=False)
            data = data.iloc[self.selected_indices_,:]
            data.index = pd.Index(range(len(data)))
            target = target.iloc[self.selected_indices_,:]
            target.index = pd.Index(range(len(target)))
            return data, target
        elif self.type == 'margins':
            Nhalfcut = self.Ncut_/2
            data = data[Nhalfcut:Nsamples-Nhalfcut]
            data.index = pd.Index(range(len(data)))
            target = target[Nhalfcut:Nsamples-Nhalfcut]
            target.index = pd.Index(range(len(target)))
            return data, target
        elif self.type == 'top':
            data = data[self.Ncut_:]
            data.index = pd.Index(range(len(data)))
            target = target[self.Ncut_:]
            target.index = pd.Index(range(len(target)))
            return data, target
        elif self.type == 'bottom':
            data = data[:Nsamples-self.Ncut_]
            data.index = pd.Index(range(len(data)))
            target = target[:Nsamples-self.Ncut_]
            target.index = pd.Index(range(len(target)))
            return data, target
        else:
            raise ValueError("Not a valid type")

class Uniformer(object):
    """ select a uniform size of groups of target values
    
    Parameters
    ----------
    bins: int or sequence of scalars or float, to be passed to pandas.cut
        If bins is an int, it defines the number of equal-width bins in the range of x. 
        However, in this case, the range of x is extended by 0.1% on each side to include 
        the min or max values of x. If bins is a sequence it defines the bin edges allowing
        for non-uniform bin width. No extension of the range of x is done in this case.
        If bins is a float, it defines the width of bins.
    
    right: bool, optional, default True
        Indicates whether the bins include the rightmost edge or not. 
        If right == True (by default), then the bins [1,2,3,4] indicate (1,2], (2,3], (3,4].
    
    include_lowest: bool, optional, default False
        Whether the first interval should be left-inclusive or not.
    
    bin_pop: int or float, optional, default 0.5
        bin_pop defines the maximum population of selected samples from each group(bin).
        If bin_pop is an int, it defines the maximum number of samples to be drawn from each group.
        A float value for bin_pop defines the fraction of the maximum population of groups as the 
        maximum size of selections from each group.
        
    substitute: str('mean','lower,'upper') or sequence of scalars, optional, default None
        If substitute is one of the choices of 'mean', 'lower' or 'upper' strings, 
        target values will be substitute with mean bin edges, lower bin edge or 
        upper bin edge, respectively. If bins is a sequence, it defines the target 
        value for bins. If None, no substitute will happen and original target 
        values would be passed out.
    
    Attributes
    ----------
    groups_: dataframe, shape (n_targets, n_bins, n_ranges)
        return groups info: target values, bin labels, range of bins
    
    grouped_indices_: dictionary
        A dict whose keys are the group labels and corresponding values being the 
        axis labels belonging to each group.
    
    selected_indices_: list
        axis labels of data points that are drawn.
    
    Returns
    -------
    data and target
    """
    def __init__(self, bins, bin_pop = 0.5, right = True, include_lowest = True,
                 substitute = None):
        self.bins = bins
        self.bin_pop = bin_pop
        self.right = right
        self.include_lowest = include_lowest
        self.substitute = substitute
        
    def fit_transform(self, data, target):
        """
        Fit the uniformer on df.
        """
        # pandas.cut
        col = target.columns[0]
        if type(self.bins) == int:
            bined = pd.cut(target[col], self.bins, right = self.right, retbins=True, labels=False,include_lowest=self.include_lowest)
        elif type(self.bins) == float:
            bins = int(max(target[col]) - min(target[col]) / self.bins)
            bined = pd.cut(target[col], bins, right = self.right, retbins=True, labels=False,include_lowest=self.include_lowest)
        else:
            bined = pd.cut(target[col], self.bins, right = self.right, retbins=True, labels=False,include_lowest=self.include_lowest)
        
        if self.right:
            ranges = ['(%f,%f]'%(bined[1][i],bined[1][i+1]) for i in bined[0]]
            if self.include_lowest:
                ranges[0] = '[' + ranges[0][1:]
        else:
            ranges = ['(%f,%f)'%(bined[1][i],bined[1][i+1]) for i in bined[0]]
            if self.include_lowest:
                ranges[0] = '[' + ranges[0][1:]
        
        # pandas.groupby
        self.groups_ = pd.DataFrame()
        self.groups_['target'] = target[col]
        self.groups_['bins'] = bined[0]
        self.groups_['ranges'] = ranges
        self.groups_.sort_values('bins',axis=0,inplace=True)
        self.grouped_indices_ = self.groups_.groupby('bins').groups
         
        if type(self.bin_pop) == int:
            pop = self.bin_pop    
        elif type(self.bin_pop) == float:
            pop = int(self.bin_pop * max([len(self.grouped_indices_[i]) for i in self.grouped_indices_]))
        else:
            raise ValueError("Wrong format of bin_pop: must be int or float")              
        
        self.selected_indices_ = _group_parser(self.grouped_indices_, pop)
        
        data = data.iloc[self.selected_indices_,:]
        data.index = pd.Index(range(len(data)))        
        if self.substitute == None :
            target = target.iloc[self.selected_indices_,:]        
            target.index = pd.Index(range(len(target)))
        else:
            if self.substitute == 'lower':
                new_target_values = [bined[1][bined[0][i]] for i in self.selected_indices_]
            elif self.substitute == 'upper':
                new_target_values = [bined[1][bined[0][i]+1] for i in self.selected_indices_]    
            elif self.substitute == 'mean':
                new_target_values = [np.mean([bined[1][bined[0][i]], bined[1][bined[0][i]+1]])for i in self.selected_indices_]
            else:
                new_target_values = list(self.substitute)
        
            target = pd.DataFrame(new_target_values, columns=[col])        
        
        return data, target

