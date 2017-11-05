=============================
Input File Generator
=============================

The ChemML wrapper's graphical user interface (GUI) facilitate the generation of the input files.
You can run GUI locally in the Jupyter notebook with two lines of python code:

.. code:: python

    from cheml import wrapperGUI
    ui = wrapperGUI()


Requirements:
    - Jupyter notebook
        * installation: http://jupyter.org/install.html
    - ipywidgets
        * installation: https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/user_install.md
    - graphviz
        * installation: https://graphviz.readthedocs.io/en/stable/manual.html#installation

Using graphviz library, you will see a graphical visualization of the workflow simultaneously.