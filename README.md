# ChemML
ChemML is a machine learning and informatics program suite for the chemical and materials sciences.


## Code Design:
ChemML is developed in the Python 2 programming language and makes use of a host of data analysis and ML libraries(accessible through the Anaconda distribution), as well as domain-specific libraries.The development follows a strictly modular and object-oriented design to make the overall code as flexible and versatile as possible.

The package consists of two Python frameworks:

- ChemML Library (cheml):
   It is a host for all the methods that are developed or coded from scratch by developers. The format of library is similar to the well known libraries like Scikit-learn.

- ChemML wrapper:
   It is an interface for many of the libraries (including cheml) that supply methods for the representation, preprocessing, analysis, mining, and modeling of large-scale chemical data sets.

## Contributors:

- Mojtaba Haghighatlari, CBE department, SUNY Buffalo
- Ramachandran Subramanian, CSE department, SUNY Buffalo
- Bhargava Urala, CSE department, SUNY Buffalo
- Gaurav Vishwakarma, CBE department, SUNY Buffalo
- Aditya Sonpal, CBE department, SUNY Buffalo
- Po-Han Chen, CBE department, SUNY Buffalo
- Srirangaraj Setlur, CSE department, SUNY Buffalo
- Johannes Hachmann, CBE department, SUNY Buffalo

- We encourage any contributions and feedback. Feel free to fork and make pull-request to the "development" branch.


## Dependencies:
ChemML only depends on numerical python (NumPy) and python data analysis (pandas) libraries. All the software and libraries
that are available through ChemML Wrapper will be imported on demand. Thus, the user is responsible for installation and providing licenses (if required).

## Citation:
Please cite the use of ChemML as:


    Haghighatlari M, Subramanian R, Urala B, Vishwakarma G, Sonpal A, Chen P, Setlur S, Hachmann J (2017) "ChemML: A machine learning and informatics program suite for the chemical and materials sciences" https://github.com/hachmannlab/chemml


## License:
ChemML is open and freely shared with the community under 3-clause BSD license.




