from __future__ import print_function
from builtins import range
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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



# class Scatter_2D(object):
#     """
#     (Scatter_2D)
#     Uses matplotlib.pyplot to create a basic 2 dimensional plot with x data and y data.
#     (https://matplotlib.org/2.2.0/index.html)
#
#     Parameters:
#     ----------
#     xheader,yheader: string or list of strings, optional (default = ['x'],['y'])
#         string or list of strings containing the names of column headers of the dataframe that needs to be plot.
#
#     nod: integer, optional (default = 1)
#         number of data sets that need to be plot
#
#     dflist : pandas dataframe or list of pandas dataframes
#         specify the data that needs to be plot
#         provide this while calling the plot function
#
#     NOTE: x data and y data in the same dataframe ( specify xheaders and yheaders )
#         x data and y data in different
#         if dflist has 1 dataframe nod can be n; for example (dflist=[df1] ; nod = 1)
#         if dflist has n dataframes nod has to be n/2  (dflist=[df1,df2,df3,df4] ; nod=2)
#
#
#     legend: Boolean, optional (default=False)
#         True if a legend is required
#
#     xmin, xmax, ymin, ymax = integer, optional (default = 0)
#         set upper and lower limits of x and y axis in the plot
#
#     title: string, optional (default= 'Plot')
#         set the figure title
#
#     xlabel, ylabel : string, optional (default = 'x','y')
#         label x and y axes
#
#     sc : list of strings, optional (default = '')
#         mention the shape and color of the points on the plot, for example 'ro', 'b+', etc.
#
#     legend_title : list of strings, optional (default = [])
#         label each plot on the legend
#
#     l_pos : string, optional (default = 'Best')
#         specify the position of the legend.
#
#     kwargs : dictionary, optional (default = {})
#         add any matplotlib options in the form of a dictionary.
#         provide keys in the form of a string.
#         for example kwargs = {'key':value}
#
#
#
#     Example:
#     --------
#     >>> from chemml.datasets import load_organic_density
#     >>> smiles,density,features=load_organic_density()
#     >>> pl3=Scatter_2D(['MW'],['AMW'],nod=1,xlabel='MW',ylabel='AMW',sc=['bo'],title='MW vs AMW', kwargs={'markersize':0.5,'alpha':0.25},xmin=800,xmax=2000,ymin=800,ymax=2000)
#     >>> fig=pl3.plot(features)
#     >>> # fig.show()
#
#     """
#     def __init__(self, xheader=['x'], yheader=['y'],legend=False,xmin=0,xmax=0,ymin=0,ymax=0,xlabel= 'x', ylabel='y',title='Plot',sc=[''],legend_titles=[],l_pos='best',kwargs={}):
#
#         self.l_pos = l_pos
#         self.legend=legend
#         self.legend_titles=legend_titles
#         self.sc=sc
#         self.xmin=xmin
#         self.xmax=xmax
#         self.ymin=ymin
#         self.ymax=ymax
#         self.xlabel = xlabel
#         self.ylabel = ylabel
#         self.title = title
#         self.kwargs=kwargs
#         self.xheader=xheader
#         self.yheader=yheader
#
#     def __fit(self,i,j):
#
#         if j==1:
#             if len(self.dfx)==1:
#                 dfx=self.dfx
#                 for i in range(len(self.dfy)-1):
#                     dfx.append(self.dfx[0])
#                 self.dfx=dfx
#             elif len(self.dfy)==1:
#                 dfy=self.dfy
#                 for i in range(len(self.dfx)-1):
#                     dfy.append(self.dfy[0])
#                 self.dfy=dfy
#             else:
#                 msg="length of dataframe lists (dfx and dfy) should be same, or one of them should be equal to 1"
#                 raise IOError(msg)
#
#         elif j==2:
#             if len(self.dfx)==1:
#                 dfx=self.dfx
#                 for i in range(len(self.xheader)-1):
#                     dfx.append(self.dfx[0])
#                 self.dfx=dfx
#             elif len(self.xheader)==1:
#                 xheader=self.xheader
#                 for i in range(len(self.dfx)-1):
#                     xheader.append(self.xheader[0])
#                 self.xheader = xheader
#             else:
#                 msg = "length of header list (xheader) should be same as dataframe list (dfx) or one of them should be equal to 1"
#                 raise IOError(msg)
#
#         elif j==3:
#             if len(self.dfy)==1:
#                 dfy=self.dfy
#                 print 'here'
#                 for i in range(len(self.yheader)-1):
#                     dfy.append(self.dfy[0])
#                 self.dfy=dfy
#             elif len(self.yheader)==1:
#                 yheader=self.yheader
#                 for i in range(len(self.dfy)-1):
#                     yheader.append(self.yheader[0])
#                 self.yheader = yheader
#             else:
#                 msg = "length of header list (yheader) should be same as dataframe list (dfy) or one of them should be equal to 1"
#                 raise IOError(msg)
#         elif j==4:
#             if len(self.xheader)==1:
#                 xheader=self.xheader
#                 for i in range(len(self.yheader)-1):
#                     xheader.append(self.xheader[0])
#                 self.xheader=xheader
#             elif len(self.yheader)==1:
#                 yheader=self.yheader
#                 for i in range(len(self.xheader)-1):
#                     yheader.append(self.yheader[0])
#                 self.yheader=yheader
#             else:
#                 msg="length of header lists (xheader and yheader) should be the same or one of them should be equal to 1"
#                 raise IOError(msg)
#
#         else:
#             dfx=self.dfx[i]
#             dfy=self.dfy[i]
#             if isinstance(self.xheader[i], str):
#                 self.x = dfx[self.xheader[i]]
#             else:
#                 self.x = dfx.iloc[:, self.xheader[i]]
#
#             if isinstance(self.yheader[i],str):
#                 self.y = dfy[self.yheader[i]]
#             else:
#                 self.y = dfy.iloc[:, self.yheader[i]]
#
#     def plot(self,dfx,dfy):
#         """
#         (plot)
#         This is the main function to create a 2D plot from dflist
#
#         Parameters:
#         ----------
#         dflist : pandas dataframe or list of pandas dataframes
#             specify the data that needs to be plot.
#             provide this while calling the plot function
#
#         Returns:
#         ------
#         f : matplotlib object
#             returns the matplotlib object containing the plot information
#
#         """
#
#         self.dfx=dfx
#         self.dfy=dfy
#
#         if len(self.xheader)!=len(self.dfx):
#             self.__fit(-1,2)
#
#         if len(self.yheader)!=len(self.dfy):
#             self.__fit(-1,3)
#
#         if len(self.dfx)!=len(self.dfy):
#             self.__fit(-1, 1)
#
#         if len(self.xheader)!=len(self.yheader):
#             self.__fit(-1,4)
#
#         if len(self.dfx)==1:
#             f=plt.figure()
#             self.__fit(0,0)
#             if self.sc==['']:
#                 plt.plot(self.x,self.y,**self.kwargs)
#             else:
#                 plt.plot(self.x,self.y,self.sc[0],**self.kwargs)
#
#         else:
#             f, ax = plt.subplots()
#             for i in range(len(self.dfx)):
#                 self.__fit(i,0)
#                 x1=self.x
#                 y1=self.y
#                 if self.sc[i]!='':
#                     plt.plot(x1,y1,self.sc[i],**self.kwargs)
#                 else:
#                     plt.plot(x1,y1,**self.kwargs)
#
#         plt.xlabel(self.xlabel)
#         plt.ylabel(self.ylabel)
#         plt.title(self.title)
#
#         if self.xmin+self.xmax!=0 and self.ymin+self.ymax!=0:
#             plt.axis([self.xmin,self.xmax,self.ymin,self.ymax])
#         if self.legend==True:
#             plt.legend(self.legend_titles,loc=self.l_pos)
#         return f

# class hist(object):
#     """
#     (hist)
#     Uses matplotlib.pyplot to create a simple histogram.
#     (https://matplotlib.org/2.2.0/index.html)
#
#     Parameters:
#     ----------
#     nbins: integer
#         number of bins
#
#     rwidth: integer, optional (default=1)
#         width of the bars in the histogram
#
#     xmin,xmax: integer, optional(default = 0,0)
#         set upper and lower limits of the x axis
#
#     bestfit: Boolean, optional (default = False)
#
#     linshapecolor: string, optional (default = '')
#         shape and color of the lines in the histogram
#
#     xlabel, ylabel: strings, optional (default = 'x','y')
#         label for the x and y axes
#
#     title : string, optional (default = 'histogram')
#         title for the figure
#
#     isformatter: Boolean, optional (default= False)
#
#     formatter_type: string, optional (default = '')
#
#     kwargs: dictionary, optional (default ={})
#         add any matplotlib options in the form of a dictionary.
#         provide keys in the form of a string.
#         for example kwargs = {'key':value}
#
#     Example:
#     --------
#     >>> from chemml.datasets import load_organic_density
#     >>> smiles,density,features=load_organic_density()
#     >>> from chemml.visualization import hist
#     >>> pl4=hist(nbins= 20, kwargs={'normed':True,'facecolor':'blue', 'ec':'black'}, lineshapecolor='g-', formatter_type='percent', rwidth=0.8, xmin=800, xmax=2000)
#     >>> fig1=pl4.plot(density)
#
#     """
#     def __init__(self,nbins,rwidth=1,xmin=0,xmax=0,bestfit=False,lineshapecolor='',xlabel='x',ylabel='y',title='histogram',isformatter=False,formatter_type='',kwargs={}):
#         self.nbins=nbins
#         self.rwidth=rwidth
#         self.xmin=xmin
#         self.lineshapecolor=lineshapecolor
#         self.bestfit=bestfit
#         self.xmax=xmax
#         self.title=title
#         self.isformatter=isformatter
#         self.formatter_type=formatter_type
#         self.xlabel=xlabel
#         self.ylabel=ylabel
#         self.kwargs=kwargs
#         self.mean=[]
#         self.std=[]
#
#     def __fit(self,df):
#         mean=[]
#         std=[]
#         self.x=np.asarray(df)
#         df.mean()
#         dfnew=df.describe()
#         mean.append(dfnew.loc['mean'])
#         std.append(dfnew.loc['std'])
#         self.mean=mean
#         self.std=std
#
#         print self.mean
#         print self.std
#
#
#     def plot(self,dflist):
#         """
#         (plot)
#         This is the main function to create a simple histogram from dflist
#
#         Parameters:
#         ----------
#         dflist : pandas dataframe or list of pandas dataframes
#             specify the data that needs to be plot.
#             provide this while calling the plot function
#
#         Returns:
#         ------
#         f : matplotlib object
#             returns the matplotlib object containing the plot information
#
#         """
#         if not isinstance(dflist, list)==True:
#             self.__fit(dflist)
#             self.dflist=[dflist]
#         else:
#             for i in dflist:
#                 self.__fit(i)
#             self.dflist=dflist
#
#         f,ax=plt.subplots()
#
#         if self.isformatter==True:
#             if self.formatter_type=='percent':
#                 formatter=matplotlib.ticker.EngFormatter()
#                 ax.yaxis.set_major_formatter(formatter)
#
#         if (self.xmin+self.xmax)!=0:
#             n,bins, patches=plt.hist(self.x,bins=self.nbins,rwidth=self.rwidth,range=(self.xmin,self.xmax),**self.kwargs)
#         else:
#             n,bins, patches=plt.hist(self.x,bins=self.nbins,rwidth=self.rwidth,**self.kwargs)
#
#
#         if self.bestfit==True and self.lineshapecolor!='':
#             y = mlab.normpdf(bins)#, self.mean, self.std)
#             plt.plot(bins,y,self.lineshapecolor,**self.kwargs)
#
#         plt.xlabel(self.xlabel)
#         plt.ylabel(self.ylabel)
#         plt.title(self.title)
#         return f
