.. _PyScript:

PyScript
============

:task:
    | Script

:host:
    | cheml

:function:
    | PyScript

:parameters:
    | li (these parameters will be sorted to define the order of compilation of python code)
    |
    .. note:: Each parameter shows a line of python code in a string format. The parameter names can be any thing and they will be sorted to define the order of compiling lines.

:send tokens:
    | ``df_out1`` : any pandas data frame
    | ``df_out2`` : any pandas data frame
    | ``api_out1`` : any instance of a class
    | ``api_out2`` : any instance of a class
    | ``var_out1`` : any other variables
    | ``var_out2`` : any other variables


:receive tokens:
    | ``df1`` : pandas data frame
    | ``df2`` : pandas data frame
    | ``api1`` : any instance of a class
    | ``api2`` : any instance of a class
    | ``var1`` : any other variables
    | ``var2`` : any other variables

:requirements:
    | :py:mod:`cheml`, version: 1.3.1

:input file view:
    | ``## Script``
    |   ``<< host = cheml``
    |   ``<< function = PyScript``
    |   ``<< l1 = "print 'df1:', df1"``
    |   ``>> id df1    >> id df2``
    |   ``>> id api1    >> id api2``
    |   ``>> id var1    >> id var2``
    |   ``>> df_out1 id    >> df_out2 id``
    |   ``>> api_out1 id    >> api_out2 id``
    |   ``>> var_out1 id    >> var_out2 id``
