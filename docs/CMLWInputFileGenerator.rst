.. _ChemML_Wrapper_InFileGen:

==============
Input File GUI
==============

...::: ChemML Wrapper is currently only available in the version 0.4.* (Python 2.7) :::...

The ChemML wrapper's graphical user interface (GUI) facilitate the generation of the input files.
You can run GUI locally in the Jupyter notebook with two lines of python code:

.. code:: python

    from cheml.notebooks import wrapperGUI
    ui = wrapperGUI()


Requirements:
    - Jupyter notebook
        * installation: http://jupyter.org/install.html
    - ipywidgets and widgetsnbextension
        * installation: https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/user_install.md
        * ipywidgets and widgetsnbextension will be installed accompanied by chemml via pip.
    - graphviz
        * installation: https://graphviz.readthedocs.io/en/stable/manual.html#installation
        * Using graphviz library, you will see a graphical visualization of your project's workflow simultaneously.
        * graphviz will be installed accompanied by chemml via pip.

We recommend downloading and installing Anaconda for python 2. This way Jupyter will be installed automatically.
If you are using anaconda and you plan to use a virtual environment, please run the following commands to
install ChemML and wrapperGUI (the first and third lines are unnecessary if you have already installed chemml):

.. code:: bash

   conda create --name my_chemml_env python=2.7
   source activate my_chemml_env
   pip install chemml --user -U

   jupyter nbextension install --py widgetsnbextension --user
   jupyter nbextension enable --sys-prefix --py widgetsnbextension
   conda install -c conda-forge nb_conda_kernels

The last command installs nb_conda_kernels, which provides a seprate Jupyter kernel for each conda environment. You need it to run
a Jupyter notebook with a kernel pointing to 'my_chemml_env' environment.

To run a notebook, you just need to run the following command in the Terminal:

.. code:: bash

   jupyter notebook

This will consequently open a notebook dashboard in your browser.
Now if you click on the 'New' button in the top right corner and select the
'python: my_chemml_env', an empty notebook will be opened in a new tab.
Please type the two above-mentioned lines of python code
and press Ctrl+Enter to run the wrapperGUI.



A link to the web application of this GUI will be posted here soon.