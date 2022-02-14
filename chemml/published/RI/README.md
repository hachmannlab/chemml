# A Physics-Infused Deep Learning Model for the Prediction of Refractive Indices and Large-Scale Screening of Organic Compound Space
This directory contains architectures and trained models reported in [this paper](https://chemrxiv.org/engage/chemrxiv/article-details/60c742e7702a9b1d4218a4f1).

## Latest Version:
   - to find out about the latest version and release history, click [here](https://pypi.org/project/chemml/#history)

Here is a list of external libraries that will be installed with chemml:
   - numpy
   - pandas
   - tensorflow
   - scikit-learn
   - matplotlib
   - seaborn
   - lxml

Since conda installation is not available for ChemML yet, we recommend installing rdkit and openbabel (please install openbabel 2.x not openbabel 3.x) in a conda virtual environment prior to installing ChemML. For doing so, you need to follow the conda installer:

    conda create --name my_chemml_env python=3.6
    source activate my_chemml_env
    conda install -c conda-forge rdkit openbabel
    pip install chemml

## Citation:
Please cite the use of ChemML as:

    Main citation:

    @article{chemml2019,
    author = {Haghighatlari, Mojtaba and Vishwakarma, Gaurav and Altarawy, Doaa and Subramanian, Ramachandran and Kota, Bhargava Urala and Sonpal, Aditya and Setlur, Srirangaraj and Hachmann, Johannes},
    journal = {ChemRxiv},
    pages = {8323271},
    title = {ChemML: A Machine Learning and Informatics Program Package for the Analysis, Mining, and Modeling of Chemical and Materials Data},
    doi = {10.26434/chemrxiv.8323271.v1},
    year = {2019}
    }

    
    Other references:

    @article{chemml_review2019,
    author = {Haghighatlari, Mojtaba and Hachmann, Johannes},
    doi = {https://doi.org/10.1016/j.coche.2019.02.009},
    issn = {2211-3398},
    journal = {Current Opinion in Chemical Engineering},
    month = {jan},
    pages = {51--57},
    title = {Advances of machine learning in molecular modeling and simulation},
    volume = {23},
    year = {2019}
    }



