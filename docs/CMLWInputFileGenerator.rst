.. _ChemML_Wrapper_InFileGen:

==============
Input File GUI
==============

The ChemML wrapper's graphical user interface (GUI) facilitates the generation of input files.

To install the dependencies for ChemML Wrapper GUI, activate the virtual environment and run the following commands:

.. code:: bash

   conda install nb_conda_kernels
   pip install graphviz
   pip install openpyxl

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

To run a notebook, you just need to run the following command in the Terminal:

.. code:: bash

   jupyter notebook

Run the GUI locally in the Jupyter notebook via: 

.. code:: python

    from chemml.wrapper.notebook import ChemMLNotebook
    ui = ChemMLNotebook()