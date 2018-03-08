import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from matplotlib.ticker import FuncFormatter
import matplotlib


class xy_plot(object):
    def __init__(self, dflist, xheader=['x'], yheader=['y'], nod=1,subplots=[1,1,1],legend=False,xmin=0,xmax=0,ymin=0,ymax=0,xlabel= 'x', ylabel='y',title='Plot',sc=[' '],legend_titles=[],l_pos='best',**kwargs):
        """
        nod=number of data sets in a plot
        subplots=[number of rows,number of columns, number of plots]

        """
        self.l_pos=l_pos
        self.dflist=[]
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
        self.dflist=dflist
        self.xheader=xheader
        self.yheader=yheader

#        self.plot(**kwargs)

    def fit(self,i):
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

    def plot(self,**kwargs):
        if self.nod==1:
            f=plt.figure()
            self.fit(self.nod)
            plt.plot(self.x,self.y,self.sc[1],**kwargs)
        elif self.nod>1:
            f, ax = plt.subplots()
            for i in range(1,self.nod+1):
                self.fit(i)
                x1=self.x
                y1=self.y
                if len(self.sc)>1 and self.sc[0]!=' ':
                    plt.plot(x1,y1,self.sc[i-1],**kwargs)
                else:
                    plt.plot(x1,y1,**kwargs)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

        if self.xmin+self.xmax!=0 and self.ymin+self.ymax!=0:
            plt.axis([self.xmin,self.xmax,self.ymin,self.ymax])
        if self.legend==True:
            plt.legend(self.legend_titles,loc=self.l_pos)
        return plt

class hist(object):
    def __init__(self,dflist,nbins,rwidth=1,xmin=0,xmax=0,bestfit=False,lineshapecolor='',xlabel='x',ylabel='y',title='histogram',isformatter=False,formatter_type='',**kwargs):
        self.dflist=dflist
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

    def fit(self):
        self.x=np.asarray(self.dflist)
        dfnew=self.dflist.describe()
        self.mean=dfnew['mean']
        self.std=dfnew['std']


    def plot(self,**kwargs):
        self.fit()
        f,ax=plt.subplots()

        if self.isformatter==True:
            if self.formatter_type=='percent':
                formatter=matplotlib.ticker.EngFormatter()
                ax.yaxis.set_major_formatter(formatter)

        if (self.xmin+self.xmax)!=0:
            n,bins, patches=plt.hist(self.x,bins=self.nbins,rwidth=self.rwidth,range=(self.xmin,self.xmax),**kwargs)
        else:
            n,bins, patches=plt.hist(self.x,bins=self.nbins,rwidth=self.rwidth,**kwargs)


        if self.bestfit==True and self.lineshapecolor!='':
            y = mlab.normpdf(bins, self.mean, self.std)
            plt.plot(bins,y,self.lineshapecolor)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        return plt
