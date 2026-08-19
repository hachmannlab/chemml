import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np



class decorator(object):
    """
    This class provides options to decorate a plot with more information.

    Parameters
    ----------
    title: string, optional (default='')
        title
    xlabel: string, optional (default='')
        the x axis label
    ylabel: string, optional (default='')
        the y axis label
    xlim: tuple, optional (default=(None, None))
        a tuple of min and max of x axis
    ylim: tuple, optional (default=(None, None))
        a tuple of min and max of y axis
    grid: boolean, optional (default=True)
        axes grids can be on (True) or off (False)
        check: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.grid.html
    gridcolor: string, optional (default='k')
        set grid color
        check this link: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
    grid_linewidth: float, optional (default=2)
        set grid line width
        check this link: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linewidth

    Examples
    --------
    >>> from chemml.visualization import hist, decorator
    >>> from chemml.datasets import load_organic_density
    >>> smiles, density, features = load_organic_density()
    >>> hg = hist(20,'g',{'normed':True})
    >>> fig = hg.plot(features,'AMW')   # atomic molecular weights
    >>> dec = decorator('histogram',xlabel='density', ylabel='%', xlim=(4,None), ylim=(0,None),\
            grid=True, grid_color='g', grid_linestyle=':', grid_linewidth=0.5)
    >>> fig = dec.fit(fig)
    >>> fig.show()
    """
    def __init__(self, title='', xlabel='', ylabel='', xlim=(None, None), ylim=(None, None),
                 grid=True, grid_color='k', grid_linestyle='--', grid_linewidth=0.5):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.grid = grid
        self.grid_color = grid_color
        self.grid_linestyle = grid_linestyle
        self.grid_linewidth = grid_linewidth

    def fit(self, figure):
        """
        the main function to fit the parameters to the input figure

        Parameters
        ----------
        figure: matplotlib.figure.Figure object or matplotlib.AxesSubplot object
            this function only proceed with one set of axes in the figure.

        Returns
        -------
        matplotlib.figure.Figure object

        """
        if str(type(figure)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
            ax = figure
            figure = figure.figure
        elif str(type(figure)) == "<class 'matplotlib.figure.Figure'>":
            ax = figure.axes[0]
        else:
            msg = 'object must be a matplotlib.AxesSubplot or matplotlib.Figure object'
            raise TypeError(msg)
        if len(figure.axes) != 1:
            msg = 'matplotlib.figure object includes more than one axes'
            raise TypeError(msg)
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.grid(color=self.grid_color, linestyle=self.grid_linestyle, linewidth=self.grid_linewidth)
        ax.grid(self.grid)
        return figure

    def matplotlib_font(self, family='DejaVu Sans', size=18, weight='normal', style='normal', variant='normal'):
        """
        The matplotlib_font function sets custom font properties.

        Notes
        -----
        Changing these parameters wil affect all the plots in your working session.

        Parameters
        ----------
        family: string, optional (default = 'normal')
            check this example: https://matplotlib.org/examples/pylab_examples/fonts_demo.html
        size: integer or string, optional (default = 18)
            check this example: https://matplotlib.org/examples/pylab_examples/fonts_demo.html
        weight: string, optional (default = 'normal')
            check this example: https://matplotlib.org/examples/pylab_examples/fonts_demo.html
        style: string, optional (default = 'normal')
            check this example: https://matplotlib.org/examples/pylab_examples/fonts_demo.html
        variant: string, optional (default = 'normal')
            check this example: https://matplotlib.org/examples/pylab_examples/fonts_demo.html

        Returns
        -------

        """
        matplotlib.rcParams.update(
            {'font.size': size, 'font.weight': weight, 'font.family': family, 'font.style': style,
             'font.variant': variant})

class scatter2D(object):
    """
    The scatter 2D plotting interface. It is built on top of the matplotlib.pyplot.plot function.

    Parameters
    ----------
    color: string, optional (default='b')
        set color
        check this link Notes for available options: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

    marker: string, optional (default='.')
        set marker
        check this link Notes for available options: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

    linestyle: string, optional (default='')
        set line style
        check this link: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
    linewidth: float, optional (default=2)
        set line width
        check this link: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linewidth

    Examples
    --------
    >>> from chemml.visualization import scatter2D
    >>> from chemml.datasets import load_organic_density
    >>> smiles, density, features = load_organic_density()
    >>> sc = scatter2D('r', marker='.')
    >>> fig = sc.plot(dfx=den, dfy=fea, x=0, y=1)
    >>> fig.show()
    """

    def __init__(self, color='b', marker='.', linestyle='', linewidth=2):
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
        self.linewidth = linewidth

    def plot(self,dfx,dfy,x,y):
        """
        the main function to plot based on the input dataframes and their headers

        Parameters
        ----------
        dfx: pandas dataframe
            the x axis data
        dfy: pandas dataframe
            the y axis data
        x: string or integer, optional (default=0)
            header or position of data in the dfx
        y: string or integer, optional (default=0)
            header or position of data in the dfy

        Returns
        -------
        matplotlib.figure.Figure object

        """
        # check data
        if isinstance(x,str):
            X = dfx[x].values
        elif isinstance(x,int):
            X = dfx.iloc[:,x].values
        else:
            msg = 'x must be string for the header or integer for the postion of data in the dfx'
            raise TypeError(msg)
        
        if isinstance(y, str):
            Y = dfy[y].values
        elif isinstance(y, int):
            Y = dfy.iloc[:, y].values
        else:
            msg = 'y must be string for the header or integer for the postion of data in the dfy'
            raise TypeError(msg)

        # instantiate figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        trash = ax.plot(X,Y,color=self.color, marker=self.marker, linestyle=self.linestyle, linewidth= self.linewidth)
        return fig

class hist(object):
    """
    The histogram plotting interface. It is built on top of the matplotlib.pyplot.hist function .
    check this link: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib.pyplot.hist

    Parameters
    ----------
    bins:  integer or sequence or 'auto', optional (default = None)
        exact same parameter from matplotlib.pyplot.hist

    color: color or array_like of colors or None, optional (default = None)
        exact same parameter from matplotlib.pyplot.hist

    kwargs : dictionary, optional (default = {})
        add any matplotlib.pyplot.hist parameter in the form of a dictionary.
        check this link: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib.pyplot.hist

        provide keys in the form of a string.
        for example kwargs = {'key':value}

    Examples
    --------
    >>> from chemml.visualization import hist
    >>> from chemml.datasets import load_organic_density
    >>> smiles, density, features = load_organic_density()
    >>> hg = hist(20,'g',{'normed':True})
    >>> fig = hg.plot(features,'AMW')   # atomic molecular weights
    >>> fig.show()

    """
    def __init__(self,bins=None, color=None, kwargs={}):
        self.bins = bins
        self.color = color
        self.kwargs = kwargs

    def plot(self, dfx, x):
        """
        the main function to plot based on the input dataframe and its header

        Parameters
        ----------
        dfx: pandas dataframe
            the x axis data
        x: string or integer, optional (default=0)
            header or position of data in the dfx

        Returns
        -------
        matplotlib.figure.Figure object

        """
        # check data
        if isinstance(x, str):
            X = dfx[x].values
        elif isinstance(x, int):
            X = dfx.iloc[:, x].values
        else:
            msg = 'x must be string for the header or integer for the postion of data in the dfx'
            raise TypeError(msg)

        # instantiate figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        tash = ax.hist(X, bins= self.bins, color=self.color, **self.kwargs)
        return fig

class SavePlot(object):
    """
    Accepts a matplotlib AxesSubplot object and saves the figure with distinct options and at a specific location.
    Displays the path to the saved figure.

    Parameters:
    ----------
    filename: string
        name of the file that needs to be saved

    output_directory: string, optional (default=None)
        specify the folder where the figure needs to be saved.
        If the output directory that is specified does not exist, a new directory is created.

    format: string, optional (default='png')
        format of the figure that needs to be saved.
        Note: we recommend 'eps' for publication quality

    kwargs : dictionary, optional (default = {})
        add any matplotlib options in the form of a dictionary.
        provide keys in the form of a string.
        for example kwargs = {'key':value}
        https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.savefig

    Example:
    --------
    >>> from chemml.datasets import load_cep_homo
    >>> from chemml.visualization import SavePlot
    >>> smiles, homo = load_cep_homo()
    >>> ax = homo.plot(kind='hist')
    >>> sa=SavePlot(filename='homo',output_directory='plots',kwargs={'facecolor':'w','dpi':100,'pad_inches':0.1, 'bbox_inches':'tight'})
    >>> sa.save(obj=ax,main_directory='project')
    The Plot has been saved at:  project/plots/homo.png

    """
    def __init__(self,filename, output_directory = None, format ='png',kwargs={}):
        self.filename = filename
        self.output_directory = output_directory
        self.format = format
        self.kwargs=kwargs

    def save(self, obj, main_directory='.'):
        """
        This is the main function that saves the figure.

        Parameters:
        ----------
        obj: matplotlib.axes._subplots.AxesSubplot or matplotlib.figure.Figure
            contains information about the plot

        main_directory: string, optional (default = '.')
            specify the parent directory where the folder needs to be saved.
            
        """
        if str(type(obj)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
            obj = obj.figure
        elif str(type(obj)) == "<class 'matplotlib.figure.Figure'>":
            pass
        else:
            msg = 'object must be a matplotlib.AxesSubplot or matplotlib.Figure object'
            raise TypeError(msg)

        if self.output_directory:
            self.output_directory = os.path.join(main_directory, self.output_directory)
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
            self.file_path = '%s/%s.%s' % (self.output_directory, self.filename,self.format)

            obj.savefig(self.file_path,**self.kwargs)

        else:
            self.file_path = '%s/%s.%s' %(main_directory,self.filename,self.format)
            obj.savefig(self.file_path,**self.kwargs)
        print('The Plot has been saved at: ', self.file_path)

class ClassificationPlots(object):
    """
    Generate classification evaluation plots including confusion matrix and ROC curves with AUC.

    Parameters
    ----------
    plot_type: string, optional (default='both')
        type of plot to generate: 'confusion_matrix', 'roc', or 'both'

    figsize: tuple, optional (default=(12, 4))
        figure size as (width, height)

    cmap: string, optional (default='Blues')
        colormap for confusion matrix

    kwargs : dictionary, optional (default = {})
        additional matplotlib options in the form of a dictionary

    Examples
    --------
    >>> from chemml.visualization import ClassificationPlots
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> cp = ClassificationPlots(plot_type='both')
    >>> fig = cp.plot(y_true, y_pred)
    >>> fig.show()
    """

    def __init__(self, plot_type='both', figsize=(12, 4), cmap='Blues', kwargs={}):
        self.plot_type = plot_type
        self.figsize = figsize
        self.cmap = cmap
        self.kwargs = kwargs

    def plot(self, y_true, y_pred, y_pred_proba=None):
        """
        Generate classification plots.

        Parameters
        ----------
        y_true: array-like
            ground truth labels

        y_pred: array-like
            predicted class labels

        y_pred_proba: array-like, optional (default=None)
            predicted probabilities for ROC curve. If None, ROC plot is skipped.

        Returns
        -------
        matplotlib.figure.Figure object

        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if self.plot_type == 'confusion_matrix':
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            self._plot_confusion_matrix(ax, y_true, y_pred)
        elif self.plot_type == 'roc':
            if y_pred_proba is None:
                raise ValueError('y_pred_proba is required for ROC curve')
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            self._plot_roc(ax, y_true, y_pred_proba)
        elif self.plot_type == 'both':
            fig = plt.figure(figsize=self.figsize)
            ax1 = fig.add_subplot(121)
            self._plot_confusion_matrix(ax1, y_true, y_pred)
            ax2 = fig.add_subplot(122)
            if y_pred_proba is not None:
                self._plot_roc(ax2, y_true, y_pred_proba)
            else:
                ax2.text(0.5, 0.5, 'y_pred_proba required for ROC curve', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_xticks([])
                ax2.set_yticks([])
        else:
            raise ValueError("plot_type must be 'confusion_matrix', 'roc', or 'both'")

        return fig

    def _plot_confusion_matrix(self, ax, y_true, y_pred):
        """
        Plot confusion matrix.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes to plot on

        y_true: array-like
            ground truth labels

        y_pred: array-like
            predicted labels

        """
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap=self.cmap, interpolation='nearest')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        tick_marks = np.arange(len(np.unique(y_true)))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    def _plot_roc(self, ax, y_true, y_pred_proba):
        """
        Plot ROC curve with AUC in legend.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes to plot on

        y_true: array-like
            ground truth labels

        y_pred_proba: array-like
            predicted probabilities

        """
        y_pred_proba = np.array(y_pred_proba)
        
        # Handle binary classification
        if y_pred_proba.ndim == 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        else:
            # Handle multiclass (one-vs-rest)
            for i in range(y_pred_proba.shape[1]):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')

