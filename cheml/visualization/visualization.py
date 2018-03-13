import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from matplotlib.ticker import FuncFormatter
import matplotlib
import os

class Scatter_2D(object):
    """
    (Scatter_2D)
    Uses matplotlib.pyplot to create a basic 2 dimensional plot with x data and y data.
    (https://matplotlib.org/2.2.0/index.html)

    Parameters:
    ----------
    xheader,yheader: string or list of strings, optional (default = ['x'],['y'])
        string or list of strings containing the names of column headers of the dataframe that needs to be plot.

    nod: integer, optional (default = 1)
        number of data sets that need to be plot (determines the number of subplots)

    subplots: list of length 3, optional (default = [1,1,1])
        [number of rows, number of columns, number of plots]

    legend: Boolean, optional (default=False)
        True if a legend is required

    xmin, xmax, ymin, ymax = integer, optional (default = 0)
        set upper and lower limits of x and y axis in the plot

    title: string, optional (default= 'Plot')
        set the figure title

    xlabel, ylabel : string, optional (default = 'x','y')
        label x and y axes

    sc : list of strings, optional (default = '')
        mention the shape and color of the points on the plot, for example 'ro', 'b+', etc.

    legend_title : list of strings, optional (default = [])
        label each plot on the legend

    l_pos : string, optional (default = 'Best')
        specify the position of the legend.

    kwargs : dictionary, optional (default = {})
        add any matplotlib options in the form of a dictionary.
        provide keys in the form of a string.
        for example kwargs = {'key':value}



    Example:
    --------
    >>> from cheml.datasets import load_organic_density
    >>> smiles,density,features=load_organic_density()
    >>> pl3=Scatter_2D(['MW'],['AMW'],nod=1,xlabel='MW',ylabel='AMW',sc=['bo'],title='MW vs AMW', kwargs={'markersize':0.5,'alpha':0.25},xmin=800,xmax=2000,ymin=800,ymax=2000)
    >>> fig=pl3.plot(features)

    """
    def __init__(self, xheader=['x'], yheader=['y'], nod=1,subplots=[1,1,1],legend=False,xmin=0,xmax=0,ymin=0,ymax=0,xlabel= 'x', ylabel='y',title='Plot',sc=[''],legend_titles=[],l_pos='best',kwargs={}):
        """
        nod=number of data sets in a plot
        subplots=[number of rows,number of columns, number of plots]

        """
        self.l_pos = l_pos

        self.legend=legend
        self.legend_titles=legend_titles
        self.sc=sc
        self.nod=nod
        self.subplots=subplots
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xheader=xheader
        self.yheader=yheader
        self.kwargs=kwargs

    # self.plot(**kwargs)

    def __fit(self,i):
        if len(self.dflist)==1:
            df=self.dflist[0]
            if isinstance(self.xheader[i-1],str)==True:
                if len(df[self.xheader[i-1]])==len(df[self.yheader[i-1]]):
                    self.x=df[self.xheader[i-1]]
                    self.y=df[self.yheader[i-1]]
            elif isinstance(self.xheader[i-1],int)==True:
                if len(df.iloc[:,self.xheader[i-1]])==len(df.iloc[:,self.yheader[i-1]]):
                    self.x=df.iloc[:,self.xheader[i-1]]
                    self.y=df.iloc[:,self.yheader[i-1]]
        else:
            if len(self.dflist[i*2-2])==len(self.dflist[i*2-1]):
                self.x=self.dflist[i*2-2]
                self.y=self.dflist[i*2-1]

    def plot(self,dflist):
        """
        (plot)
        This is the main function to create a 2D plot from dflist

        Parameters:
        ----------
        dflist : pandas dataframe or list of pandas dataframes
            specify the data that needs to be plot.
            provide this while calling the plot function

        Returns:
        ------
        f : matplotlib object
            returns the matplotlib object containing the plot information

        """
        if not isinstance(dflist, list)==True:
            self.dflist=[dflist]
        else :
            self.dflist=dflist

        if self.nod==1:
            f=plt.figure()
            self.__fit(self.nod)
            if self.sc[0]=='':
                plt.plot(self.x,self.y,**self.kwargs)
            else:
                plt.plot(self.x, self.y, self.sc[0], **self.kwargs)
        elif self.nod>1:
            f, ax = plt.subplots()
            for i in range(1,self.nod+1):
                self.__fit(i)
                x1=self.x
                y1=self.y
                if len(self.sc)>1 and self.sc[0]!=' ':
                    plt.plot(x1,y1,self.sc[i-1],**self.kwargs)
                else:
                    plt.plot(x1,y1,**self.kwargs)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

        if self.xmin+self.xmax!=0 and self.ymin+self.ymax!=0:
            plt.axis([self.xmin,self.xmax,self.ymin,self.ymax])
        if self.legend==True:
            plt.legend(self.legend_titles,loc=self.l_pos)
        return f

class hist(object):
    """
    (hist)
    Uses matplotlib.pyplot to create a simple histogram.
    (https://matplotlib.org/2.2.0/index.html)

    Parameters:
    ----------
    nbins: integer
        number of bins

    rwidth: integer, optional (default=1)
        width of the bars in the histogram

    xmin,xmax: integer, optional(default = 0,0)
        set upper and lower limits of the x axis

    bestfit: Boolean, optional (default = False)

    linshapecolor: string, optional (default = '')
        shape and color of the lines in the histogram

    xlabel, ylabel: strings, optional (default = 'x','y')
        label for the x and y axes

    title : string, optional (default = 'histogram')
        title for the figure

    isformatter: Boolean, optional (default= False)

    formatter_type: string, optional (default = '')

    kwargs: dictionary, optional (default ={})
        add any matplotlib options in the form of a dictionary.
        provide keys in the form of a string.
        for example kwargs = {'key':value}

    Example:
    --------
    >>> from cheml.datasets import load_organic_density
    >>> smiles,density,features=load_organic_density()
    >>> from cheml.visualization import hist
    >>> pl4=hist(nbins= 20, kwargs={'normed':True,'facecolor':'blue', 'ec':'black'}, lineshapecolor='g-', formatter_type='percent', rwidth=0.8, xmin=800, xmax=2000)
    >>> fig1=pl4.plot(density)

    """
    def __init__(self,nbins,rwidth=1,xmin=0,xmax=0,bestfit=False,lineshapecolor='',xlabel='x',ylabel='y',title='histogram',isformatter=False,formatter_type='',kwargs={}):
        self.nbins=nbins
        self.rwidth=rwidth
        self.xmin=xmin
        self.lineshapecolor=lineshapecolor
        self.bestfit=bestfit
        self.xmax=xmax
        self.title=title
        self.isformatter=isformatter
        self.formatter_type=formatter_type
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.kwargs=kwargs
        #self.mean=[]
        #self.std=[]

    def __fit(self,df):
        self.x=np.asarray(df)
        #df.mean()
        #dfnew=df.describe()
        #self.mean.append(dfnew['mean'])
        #self.std.append(dfnew['std'])

    def plot(self,dflist):
        """
        (plot)
        This is the main function to create a simple histogram from dflist

        Parameters:
        ----------
        dflist : pandas dataframe or list of pandas dataframes
            specify the data that needs to be plot.
            provide this while calling the plot function

        Returns:
        ------
        f : matplotlib object
            returns the matplotlib object containing the plot information

        """
        if not isinstance(dflist, list)==True:

            self.__fit(dflist)
            self.dflist=[dflist]

        else :
            for i in dflist:
                self.__fit(i)
            self.dflist=dflist

        f,ax=plt.subplots()

        if self.isformatter==True:
            if self.formatter_type=='percent':
                formatter=matplotlib.ticker.EngFormatter()
                ax.yaxis.set_major_formatter(formatter)

        if (self.xmin+self.xmax)!=0:
            n,bins, patches=plt.hist(self.x,bins=self.nbins,rwidth=self.rwidth,range=(self.xmin,self.xmax),**self.kwargs)
        else:
            n,bins, patches=plt.hist(self.x,bins=self.nbins,rwidth=self.rwidth,**self.kwargs)


        if self.bestfit==True and self.lineshapecolor!='':
            y = mlab.normpdf(bins, self.mean, self.std)
            print 'lineshape'
            plt.plot(bins,y,self.lineshapecolor,**self.kwargs)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        return f

class SaveFigure(object):
    """
    (SaveFigure)
    Accepts a matplotlib object and saves the figure with distinct options and at a specific location.
    Displays the path to the saved figure.

    Parameters:
    ----------
    obj: matplotlib object
        contains information about the plot

    filename: string
        name of the file that needs to be saved

    output_directory: string, optional (default=None)
        specify the folder where the figure needs to be saved.
        If the output directory that is specified does not exist, a new directory is created.

    format: string, optional (default='png')
        format of the figure that needs to be saved.

    kwargs : dictionary, optional (default = {})
        add any matplotlib options in the form of a dictionary.
        provide keys in the form of a string.
        for example kwargs = {'key':value}

    Example:
    --------
    >>> from cheml.visualization import SaveFigure
    >>> sav=SaveFigure(fig1,'abc1','plots',kwargs={'facecolor':'w','dpi':100,'pad_inches':0.1})
    >>> sav.fit(main_directory='plots')
    The Plot has been saved at:  plots/plots/abc1.png

    """
    def __init__(self,obj, filename, output_directory = None, format ='png',kwargs={}):
        self.filename = filename
        self.output_directory = output_directory
        self.format = format
        self.obj = obj
        self.kwargs=kwargs

    def fit(self, main_directory='.'):
        """
        (fit)
        This is the main function that saves the figure.

        Parameters:
        ----------

        main_directory: string, optional (default = '.')
            specify the parent directory where the folder needs to be saved.
            
        """

        t=type(self.obj)
        plt=self.obj
        if not 'matplotlib' in str(t):
            msg = 'object must be a matplotlib object'
            raise TypeError(msg)

        if self.output_directory:
            self.output_directory = main_directory + '/' + self.output_directory
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
            self.file_path = '%s/%s.%s' % (self.output_directory, self.filename,self.format)

            plt.savefig(self.file_path,**self.kwargs)

        else:
            self.file_path = '%s/%s.%s' %(main_directory,self.filename,self.format)
            plt.savefig(self.file_path,**self.kwargs)
        print 'The Plot has been saved at: ', self.file_path