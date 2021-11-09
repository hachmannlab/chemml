.. _ChemML_Wrapper_InFileGen:

==============
Input File GUI
==============

MAC OS and Linux users may run the commands directly in the terminal once ChemML has been installed.

For Windows users, we recommend installing WSL: https://docs.microsoft.com/en-us/windows/wsl/install-win10 
       * Through the WSL terminal, install ChemML using the Python Package Index (PyPI) via pip as specified in the installation tutorial.
       * The steps that follow must be run in the WSL terminal.

The ChemML wrapper's graphical user interface (GUI) facilitates the generation of input files.

To run a notebook, the user must run the following commands in the terminal with the environment, which has ChemML installed, activated:

.. code:: bash

   jupyter notebook

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

A Jupyter notebook will open for the current environment.
In the new Jupyter notebook, ensure the Kernel for the desired environment is activated. This can be done using: 
        * Under Kernel > Change Kernel > Python (environment name)


Run the GUI locally in the Jupyter notebook via: 

.. code:: python

    from chemml.wrapper.notebook import ChemMLNotebook
    ui = ChemMLNotebook()

Instructions to run the input files, created using the Jupyter GUI, will appear once the input file is saved using the GUI.