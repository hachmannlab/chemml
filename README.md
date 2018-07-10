# ChemML
ChemML is a machine learning and informatics program suite for the analysis, mining, and modeling of chemical and materials data.


## Code Design:
ChemML is developed in the Python 2 programming language and makes use of a host of data analysis and ML libraries(accessible through the Anaconda distribution), as well as domain-specific libraries.The development follows a strictly modular and object-oriented design to make the overall code as flexible and versatile as possible.

The package consists of two Python frameworks:

- ChemML Library (cheml):
   It is a host for all the methods that are developed or coded from scratch by developers. The format of library is similar to the well known libraries like Scikit-learn.

- ChemML wrapper:
   It is an interface for many of the libraries (including cheml) that supply methods for the representation, preprocessing, analysis, mining, and modeling of large-scale chemical data sets.


## Version:
- Program Version: 0.4.2
- Release Date: March 25, 2018

## Installation and Dependencies:
You can download ChemML from Python Package Index (PyPI) via pip. The current version of ChemML only supports Python 2.7
and it can be installed on Linux and OSX operating systems.

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

   conda create --name my_chemml_env python=2.7
   source activate my_chemml_env
   pip install chemml --user -U

you can test the installation with:

    nosetests -v cheml.tests

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
    - Mojtaba Haghighatlari received a 2018 Phase-I Software Fellowship by the Molecular Sciences Software Institute (MolSSI) for his work on ChemML.


