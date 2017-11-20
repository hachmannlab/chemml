.. ChemML documentation master file, created by
   sphinx-quickstart on Thu Jun  2 13:42:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

 |Beta|

Welcome to the ChemML's documentation!
======================================
ChemML is a machine learning and informatics program suite for the chemical and materials sciences.


Code Design:
++++++++++++
ChemML is being developed in the Python programming language and makes use of a host of data analysis and ML libraries
(accessible through the Anaconda distribution), as well as domain-specific libraries.
The development follows a strictly modular and object-oriented design to make the overall code as flexible and versatile as possible.

The package consists of two python frameworks:

- ChemML library (cheml):
   It is a host for all the methods that are developed or coded from scratch by developers. The format of library is similar to the well known libraries like Scikit-learn.

- ChemML wrapper:
   It is an interface for many of the libraries (including cheml) that supply methods for the representation, analysis, mining, and modeling of large-scale chemical data sets.

Contributors:
+++++++++++++

- Mojtaba Haghighatlari, CBE department, SUNY Buffalo
- Ramachandran Subramanian, CSE department, SUNY Buffalo (Magpie's wrapper)
- Bhargava Urala, CSE department, SUNY Buffalo (Keras' wrapper)
- Gaurav Vishwakarma, CBE department, SUNY Buffalo (Deap's wrapper)
- Po-Han Chen, CBE department, SUNY Buffalo (Docker image)
- Srirangaraj Setlur, CSE department, SUNY Buffalo
- Johannes Hachmann, CBE department, SUNY Buffalo

- You can become a developer of ChemML! Feel free to fork and make pull-request to the "development" branch.

Dependencies:
+++++++++++++
ChemML only depends on numerical python (NumPy) and python data analysis (pandas) libraries. However, using other libraries
that are available through ChemML wrapper is optional and depends on the user request.

Citation:
+++++++++
Please cite the use of ChemML as:

::

    Haghighatlari M, Subramanian R, Urala B, Vishwakarma G, Chen P, Setlur S, Hachmann J (2017) "ChemML: A machine learning and informatics program suite for the chemical and materials sciences" https://bitbucket.org/hachmanngroup/cheml


.. toctree::
   :maxdepth: 2
   :caption: ChemML Wrapper documentation

   CMLWInputFile
   CMLWInputFileGenerator
   CMLWContentsTable
   CMLWReference

.. toctree::
   :maxdepth: 2
   :caption: ChemML library documentation

   cheml

License:
++++++++
ChemML is open and freely shared with the community under 3-clause BSD license.


.. |Beta| image:: http://stlth.io/images/stlth-beta.jpg
   :width: 70 px
