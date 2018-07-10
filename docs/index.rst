.. ChemML documentation master file, created by
   sphinx-quickstart on Thu Jun  2 13:42:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|Beta|

Welcome to the ChemML's documentation!
======================================
ChemML is a machine learning and informatics program suite for the chemical and materials sciences.

    - source: https://github.com/hachmannlab/chemml
    - documentation: https://hachmannlab.github.io/chemml

Code Design:
++++++++++++
ChemML is developed in the Python 2 programming language and makes use of a host of data analysis and ML libraries
(accessible through the Anaconda distribution), as well as domain-specific libraries.
The development follows a strictly modular and object-oriented design to make the overall code as flexible and versatile as possible.

The package consists of two python frameworks:

- ChemML library (cheml):
   It is a host for all the methods that are developed or coded from scratch by developers. The format of library is similar to the Scikit-learn library.

- ChemML wrapper:
   It is an interface for many of the libraries (including cheml) that supply methods for the representation, analysis, mining, and modeling of large-scale chemical data sets.
   The wrapper is not just an interface for the cheml library. It facilitates the broader dissemination of available methods/tools as they are but in a compatible environment.

Installation and Dependencies:
++++++++++++++++++++++++++++++
You can download ChemML from Python Package Index (PyPI) via pip. The current version of ChemML only supports Python 2.7
and it can be installed on Linux and OSX operating systems.

.. code:: bash

    pip install chemml --user -U

ChemML only depends on numerical python (NumPy) and python data analysis (pandas) libraries. Using other libraries
that are available through ChemML wrapper is optional and depends on the user request. However, we also install some of the libraries
that ChemML is interfacing with, only if they can be easily and freely installed via pip.

Here is a list of external libraries and their version that will be installed with chemml:
   - numpy (>=1.13)
   - pandas (>=0.20.3)
   - tensorflow (==1.1.0)
   - keras (==2.1.5)
   - scikit-learn (==0.19.1)
   - babel (>=2.3.4)
   - matplotlib (>=1.5.1)
   - deap (>=1.2.2)
   - lxml
   - ipywidgets (>=7.1)
   - widgetsnbextension (>=3.1)
   - graphviz

Since some of the dependencies are accompanied by an exact version number, we recommend installation of ChemML in a virtual environment.
If you have Anaconda installed on your system, you can enter:

.. code:: bash

   conda create --name my_chemml_env python=2.7
   source activate my_chemml_env
   pip install chemml --user -U


you can test the installation with:

.. code:: bash

    nosetests -v cheml.tests


Contributors:
+++++++++++++

- Mojtaba Haghighatlari, CBE department, SUNY Buffalo
- Ramachandran Subramanian, CSE department, SUNY Buffalo
- Bhargava Urala, CSE department, SUNY Buffalo
- Gaurav Vishwakarma, CBE department, SUNY Buffalo
- Aditya Sonpal, CBE department, SUNY Buffalo
- Po-Han Chen, CBE department, SUNY Buffalo
- Srirangaraj Setlur, CSE department, SUNY Buffalo
- Johannes Hachmann, CBE department, SUNY Buffalo

- We encourage any contributions and feedback. Feel free to fork and make pull-request to the "development" branch.


Citation:
+++++++++
Please cite the use of ChemML as:

::

    Haghighatlari M, Subramanian R, Urala B, Vishwakarma G, Sonpal A, Chen P, Setlur S, Hachmann J (2017) "ChemML: A machine learning and informatics program suite for the chemical and materials sciences" https://github.com/hachmannlab/chemml


License:
++++++++
ChemML is open and freely shared with the community under modified 3-clause BSD license.


.. toctree::
   :maxdepth: 2
   :caption: ChemML Wrapper documentation

   CMLWTutorial
   CMLWInputFile
   CMLWInputFileGenerator
   CMLWContentsTable
   CMLWReference
..   CMLWInputFileTutorial
..   CMLWInputFileTemplates


.. toctree::
   :maxdepth: 2
   :caption: ChemML library documentation

   cheml


.. |Beta| image:: http://stlth.io/images/stlth-beta.jpg
   :width: 70 px
   :target: https://mojtabah.github.io/ChemML
