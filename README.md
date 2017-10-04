# ChemML
ChemML is a machine learning and informatics program suite for the chemical and materials sciences.


## Code Design:
ChemML is being developed in the Python programming language and makes use of a host of data analysis and ML libraries(accessible through the Anaconda distribution), as well as domain-specific libraries.The development follows a strictly modular and object-oriented design to make the overall code as flexible and versatile as possible.

The package consists of two Python frameworks:

- ChemML Library (cheml):
   It is a host for all the methods that are developed or coded from scratch by developers. The format of library is similar to the well known libraries like Scikit-learn.

- ChemML wrapper:
   It is an interface for many of the libraries (including cheml) that supply methods for the representation, preprocessing, analysis, mining, and modeling of large-scale chemical data sets.

## Dependencies:
ChemML only depends on numerical python (NumPy) and python data analysis (pandas) libraries. All the software and libraries
that are available through ChemML Wrapper will be imported on demand. Thus, the user is responsible for installation and providing licenses (if required).

## Citation:
Please cite the use of ChemML as:


    Haghighatlari M, Hachmann J (2017) "ChemML: A machine learning and informatics program suite for the chemical and materials sciences" https://bitbucket.org/hachmanngroup/cheml

This repository is in transition from our group bitbucket repository. We will make the repository publicly visible as soon as the beta version is ready. 

## Installation
ChemML is not available for installation yet. It will be open under 3-clause BSD license.



