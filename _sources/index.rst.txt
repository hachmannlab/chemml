.. ChemML documentation master file, created by
   sphinx-quickstart on Thu Jun  2 13:42:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|logo| |Beta|

Welcome to the ChemML's documentation!
======================================
ChemML is a machine learning and informatics program suite for the analysis, mining, and modeling of chemical and materials data.

    - source: https://github.com/hachmannlab/chemml
    - documentation: https://hachmannlab.github.io/chemml

Code Design:
++++++++++++
ChemML is developed in the Python 3 programming language and makes use of a host of data analysis and ML libraries(accessible through the Anaconda distribution), as well as domain-specific libraries.
The development follows a strictly modular and object-oriented design to make the overall code as flexible and versatile as possible.

The format of library is similar to the well known libraries like Scikit-learn. ChemML is also available
via graphical user interface provided by [ChemEco](https://github.com/hachmannlab/chemeco).
ChemEco is a general-purpose framework for data mining without coding. It also interfaces with many of the libraries that supply methods for the
representation, preprocessing, analysis, mining, and modeling of large-scale chemical data sets.

Version:
++++++++
- Program Version: 0.5.1
- Release Date: April 2, 2019

Installation and Dependencies:
++++++++++++++++++++++++++++++
You can download ChemML from Python Package Index (PyPI) via pip.

.. code:: bash

    pip install chemml --user -U

Here is a list of external libraries that will be installed with chemml:
   - numpy
   - pandas
   - tensorflow
   - keras
   - scikit-learn
   - matplotlib
   - seaborn
   - lxml

Since conda installation is not available for ChemML yet, we recommend installing rdkit and openbabel in a conda virtual environment prior to
installing ChemML. For doing so, you need to follow the conda installer:

.. code:: bash

    conda create --name my_chemml_env python=3.6
    source activate my_chemml_env
    conda install -c openbabel openbabel
    conda install -c rdkit rdkit
    pip install chemml

Python 2 Fans:
++++++++++++++

The library was initially compatible with Python 2 and 3, and we still develop compatible codes. However, since the Python community
and some of the dependencies are phasing out support for Python 2, we also removed it from the classifier languages.
However, you should still be able to clone the ChemML repository from github and install the older versions of some of the dependencies that
support Python 2 prior to using ChemML locally.


Citation:
+++++++++
Please cite ChemML as follows:

::

    (1) M. Haghighatlari, J. Hachmann, ChemML – A Machine Learning and Informatics Program Suite for the Analysis, Mining, and Modeling of Chemical and Materials Data, in preparation (2018).
    (2) M. Haghighatlari, J. Hachmann, A Machine Learning and Informatics Program Suite for Chemical and Materials Data Mining. Available from: https://hachmannlab.github.io/chemml.
    (3) J. Hachmann, M.A.F. Afzal, M. Haghighatlari, Y. Pal, Building and Deploying a Cyberinfrastructure for the Data-Driven Design of Chemical Systems and the Exploration of Chemical Space, Mol. Simul. 44 (2018), 921-929. DOI: 10.1080/08927022.2018.1471692


.. toctree::
   :maxdepth: 2
   :caption: ChemML Wrapper documentation

   CMLWTutorial
   CMLWInputFile
   CMLWInputFileGenerator
   CMLWContentsTable
   CMLWReference

.. toctree::
   :maxdepth: 2
   :caption: ChemML library Tutorial

   molecule.ipynb
   active_model_based.ipynb
   ga_hyper_opt.ipynb
   ga_feature_selection.ipynb

..   tutorial.chem
..   tutorial.optimization


.. toctree::
   :maxdepth: 2
   :caption: ChemML library documentation

   chemml.chem
   chemml.chem.magpie_python
   chemml.initialization
   chemml.datasets
   chemml.preprocessing
   chemml.models
   chemml.optimization
   chemml.visualization




License:
++++++++
ChemML is copyright (C) 2014-2018 Johannes Hachmann and Mojtaba Haghighatlari, all rights reserved.
ChemML is distributed under 3-Clause BSD License (https://opensource.org/licenses/BSD-3-Clause).


About us:
++++++++

:Maintainers:
    - Johannes Hachmann, hachmann@buffalo.edu
    - Mojtaba Haghighatlari
    University at Buffalo - The State University of New York (UB)

:Contributors:
    - Ramachandran Subramanian (UB): Magpie descriptor library port
    - Gaurav Vishwakarma (UB): automated model optimization
    - Bhargava Urala Kota (UB): library database
    - Aditya Sonpal (UB): debugging
    - Srirangaraj Setlur (UB): scientific advice
    - Venugopal Govindaraju (UB): scientific advice
    - Krishna Rajan (UB): scientific advice

    - We encourage any contributions and feedback. Feel free to fork and make pull-request to the "development" branch.

:Acknowledgements:
    - ChemML is based upon work supported by the U.S. National Science Foundation under grant #OAC-1751161 and in part by #OAC-1640867.
    - ChemML was also supported by start-up funds provided by UB's School of Engineering and Applied Science and UB's Department of Chemical and Biological Engineering, the New York State Center of Excellence in Materials Informatics through seed grant #1140384-8-75163, and the U.S. Department of Energy under grant #DE-SC0017193.
    - Mojtaba Haghighatlari received a 2018 Phase-I and 2019 Phase-II Software Fellowship by the Molecular Sciences Software Institute (MolSSI) for his work on ChemML.


.. |logo| image:: images/logo.png
   :width: 140 px
   :target: https://mojtabah.github.io/ChemML

.. |Beta| image:: http://stlth.io/images/stlth-beta.jpg
   :width: 70 px
   :target: https://mojtabah.github.io/ChemML

