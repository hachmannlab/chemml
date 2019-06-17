[![Build Status](https://travis-ci.org/hachmannlab/chemml.svg?branch=master)](https://travis-ci.org/hachmannlab/chemml)
[![codecov](https://codecov.io/gh/hachmannlab/chemml/branch/master/graph/badge.svg)](https://codecov.io/gh/hachmannlab/chemml)
[![version status](http://img.shields.io/pypi/v/chemml.svg?style=flat)](https://pypi.python.org/pypi/chemml)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/hachmannlab/chemml/blob/master/LICENSE)

# ChemML
ChemML is a machine learning and informatics program suite for the analysis, mining, and modeling of chemical and materials data.
Please check the [ChemML website](https://hachmannlab.github.io/chemml) for more information.

   - ChemML documentation: https://hachmannlab.github.io/chemml

## Code Design:
ChemML is developed in the Python 3 programming language and makes use of a host of data analysis and ML libraries(accessible through the Anaconda distribution), as well as domain-specific libraries. 
The development follows a strictly modular and object-oriented design to make the overall code as flexible and versatile as possible.

The format of library is similar to the well known libraries like Scikit-learn. ChemML will be soon available 
via graphical user interface provided by [ChemEco](https://github.com/hachmannlab/chemeco).
ChemEco is a general-purpose framework for data mining without coding. It also interfaces with many of the libraries that supply methods for the 
representation, preprocessing, analysis, mining, and modeling of large-scale chemical data sets.


## Latest Version:
   - to find the latest version and release history, click [here](https://pypi.org/project/chemml/#history)

## Installation and Dependencies:
You can download ChemML from PyPI via pip.

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

    conda create --name my_chemml_env python=3.6
    source activate my_chemml_env
    conda install -c openbabel openbabel
    conda install -c rdkit rdkit
    pip install chemml

## Python 2 Fans
The library was initially compatible with Python 2 and 3, and we still develop compatible codes. However, since the Python community
and some of the dependencies are phasing out support for Python 2, we also removed it from the classifier languages.
However, you should still be able to clone the ChemML repository from github and install the older versions of some of the dependencies that 
support Python 2, prior to using ChemML locally.
 
## Citation:
Please cite the use of ChemML as:


   - M. Haghighatlari, J. Hachmann, ChemML – A Machine Learning and Informatics Program Suite for the Analysis, Mining, and Modeling of Chemical and Materials Data, in preparation (2018).
   - M. Haghighatlari, J. Hachmann, A Machine Learning and Informatics Program Suite for Chemical and Materials Data Mining. Available from: https://hachmannlab.github.io/chemml.
   - J. Hachmann, M.A.F. Afzal, M. Haghighatlari, Y. Pal, Building and Deploying a Cyberinfrastructure for the Data-Driven Design of Chemical Systems and the Exploration of Chemical Space, Mol. Simul. 44 (2018), 921-929. DOI: 10.1080/08927022.2018.1471692

## License:
ChemML is copyright (C) 2014-2018 Johannes Hachmann and Mojtaba Haghighatlari, all rights reserved.
ChemML is distributed under 3-Clause BSD License (https://opensource.org/licenses/BSD-3-Clause).

## About us:

### Maintainers:
    - Johannes Hachmann, hachmann@buffalo.edu
    - Mojtaba Haghighatlari
    University at Buffalo - The State University of New York (UB)

### Contributors:
    - Doaa Altarawy (MolSSI): scientific advice and software mentor 
    - Ramachandran Subramanian (UB): Magpie descriptor library port
    - Gaurav Vishwakarma (UB): automated model optimization
    - Bhargava Urala Kota (UB): library database
    - Aditya Sonpal (UB): debugging
    - Srirangaraj Setlur (UB): scientific advice
    - Venugopal Govindaraju (UB): scientific advice
    - Krishna Rajan (UB): scientific advice

    - We encourage any contributions and feedback. Feel free to fork and make pull-request to the "development" branch.

### Acknowledgements:
    - ChemML is based upon work supported by the U.S. National Science Foundation under grant #OAC-1751161 and in part by #OAC-1640867.
    - ChemML was also supported by start-up funds provided by UB's School of Engineering and Applied Science and UB's Department of Chemical and Biological Engineering, the New York State Center of Excellence in Materials Informatics through seed grant #1140384-8-75163, and the U.S. Department of Energy under grant #DE-SC0017193.
    - Mojtaba Haghighatlari received 2018 Phase-I and 2019 Phase-II Software Fellowships by the Molecular Sciences Software Institute (MolSSI) for his work on ChemML.


