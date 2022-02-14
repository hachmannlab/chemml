Accelerated Discovery of High-Refractive-Index Molecules
========================================================

We present a high-throughput computational study to identify novel polyimides (PIs) with exceptional refractive index (RI) values for use as optic or optoelectronic materials. Our study utilizes an RI prediction protocol based on a combination of first-principles and data modeling developed in previous work, which we employ on a large-scale PI candidate library generated with the ChemLG code. We deploy the virtual screening software ChemHTPS to automate the assessment of this extensive pool of PI structures in order to determine the performance potential of each candidate. This rapid and efficient approach yields a number of highly promising leads compounds. Using the data mining and machine learning program package ChemML, we analyze the top candidates with respect to prevalent structural features and feature combinations that distinguish them from less promising ones. In particular, we explore the utility of various strategies that introduce highly polarizable moieties into the PI backbone to increase its RI yield. The derived insights provide a foundation for rational and targeted design that goes beyond traditional trial-and-error searches.

For this study, we created a library of 1.5 million small organic molecules using an initial set of 15 building blocks (and constraints on molecular weight and number of ring-moieties per molecule) via ChemLG.

We select a random subset of 100,000 molecules from this library and perform high-throughput electronic structure theory calculations to obtain Polarizability using the Kohn-Sham density functional theory (DFT). For these molecules, the initial geometries are generated using the MMFF94s forcefield (from Openbabel software), and are subsequently optimized using BP86 generalized gradient approximation (GGA) functional with the double-:math:`{\zeta}` basis set by the Karslruhe group. In order to calculate Polarizability, we then perform single point energy calculations using an all-electron restricted DFT framework with the PBE0 hybrid functional and the same double-:math:`{\zeta}` def2-SVP basis set along with Grimme's D3 correction to account for dispersion interactions. 

Since RI is to be estimated from \pol, we additionally use a larger and more accurate triple-:math:`{\zeta}` def2-TZVP basis set from the Karlsruhe group for Polarizability. However, considering the computational overhead of these extra calculations at higher levels of quantum theory, we only use the triple-:math:`{\zeta}` basis set for a random subset of 10,000 molecules from the 100,000 molecules considered for DFT. All our DFT calculations are carried out using the ORCA 3.0.2 quantum chemistry program package \cite{Neese2012} with default settings. By default, these ORCA calculations yield polarizability volume or \pol' in :math:`{Bohr^3}`. For the purposes of this chapter, the def2-SVP and def2-TZVP basis sets and corresponding datasets are abbreviated to SVP and TZVP. 


Next, we calculate RI using these Polarizability values and Number Density (obtained from molecular dynamics simulations) and the resulting dataset serves as the ground-truth for our data modeling efforts. In order to accelerate screening of the remainder of the molecules in the library, we develop highly acccurate ML surrogates for predicting Polarizability, RI, and Number Density. In this pursuit, we use a variety of advanced deep neural network (DNN) architectures as they are promising for their performance and flexibility. We benchmark and compare these models and outline some of their strengths and weaknesses in the process. 

For all ML models, we use two families of feature representations: 

    - Molecular descriptors: 1893 two-dimensional topological and physicochemical descriptors from Dragon 7 software 
    
    - Molecular fingerprints (FP): Three sets of molecular graph based FP vectors (each of length 1024 bits) from RDKit -- Morgan FP with circular radius 2, hashed atom pair FP (HAP), hashed topological torsion FP (HTT)


A train-test split of 9:1 is used to fit the data to the DNNs with additional parameters for early stopping, dropout, and l2 regularization parameter to avoid overfitting. ChemML's genetic algorithm implementation is used to optimize the hyperparameters, such as the regularization parameter, number of neurons and hidden layers, learning rate, etc., that dictate the architecture as well as the training of DNNs. . For assessing the ML model's performance, we report the following regression metrics : mean absolute error (MAE), root mean squared error (RMSE), mean absolute percentage error (MAPE), and the coefficient of determination :math:`{R^2}`.

Single Target DNNs
++++++++++++++++++

.. code:: python

    # load SMILES representation and target properties of 100,000 molecules
    from chemml.published.RI import load_small_organic_data()

    molecules, targets = load_small_organic_data()


    # convert the SMILES representation of molecules to chemml.chem.Molecule object
    from chemml.chem import Molecule

    mol_list = [Molecule(i['smiles'], 'smiles') for i in molecules]


    # generate Morgan, HAP and HTT fingerprints
    from chemml.chem import RDKitFingerprint

    morgan = RDKitFingerprint(fingerprint_type='morgan').represent(mol_list)
    hap = RDKitFingerprint(fingerprint_type='hap').represent(mol_list)
    htt = RDKitFingerprint(fingerprint_type='htt').represent(mol_list)


    # generate Dragon descriptors
    from chemml.chem import Dragon

    dragon = Dragon().represent(mol_list)


    feature_sets = [dragon, morgan, hap, htt]
    targets = [targets['refractive_index'], targets['polarizability'], targets['number_density']]


    # generate indices for splitting the features and targets into training and test sets
    from sklearn.model_selection import train_test_split

    train_indices, test_indices = train_test_split(range(100000), test_size=0.1, random_state=13)

    # Model Training
    from sklearn.preprocessing import StandardScaler
    from chemml.utils import regression_metrics
    from chemml.models import MLP
    from chemml.published.RI import load_hyperparameters


    for f, f_name in zip(feature_sets,['dragon', 'morgan', 'hap', 'htt']):
        for t, t_name in zip(targets, ['refractive_index', 'polarizability', 'number_density']):
            
            # scalers for features and targets
            xscale, yscale = StandardScaler(), StandardScaler()
            
            # initialize a neural network object
            mlp = MLP()
            
            
            # load optimized hyperparameters into MLP object
            mlp = load_hyperparameters(mlp, f_name, t_name, 'single')
            
            # model fitting
            mlp.fit(X = xscale.fit_transform(f[train_indices]), 
                    y = yscale.fit_transform(t[train_indices]).reshape(-1,1))
            
            # model predictions on test data
            y_pred = yscale.inverse_transform(mlp.predict(xscale.transform(f[test_indices])))
            metrics_df = regression_metrics(t[test_indices], y_pred)

Physics Infused DNN 
++++++++++++++++++++

.. code:: python

    from chemml.published.RI import LorentzLorenz

    # instantiate LorentzLorenz() class object
    physics_infused_model = LorentzLorenz(n_features=4)

    # process input data according to the format required for this class
    X_train, X_test, y_train, y_test, scaler_y = physics_infused_model.preprocessing(
                                            features=features, targets=targets, return_scaler=True)

    # fit the model to training data
    physics_infused_model = physics_infused_model.fit(X_train, y_train)

    # get predictions on test data
    y_pred = physics_infused_model.predict(X_test)


Transfer learning TZVP Polarizabilities
+++++++++++++++++++++++++++++++++++++++

.. code:: python

    # load TZVP data
    from chemml.published.RI import load_small_organic_data_10k()

    molecules, targets = load_small_organic_data_10k()


    # convert the SMILES representation of molecules to chemml.chem.Molecule object
    mol_list = [Molecule(i['smiles'], 'smiles') for i in molecules]

    # generate Dragon descriptors
    dragon_tzvp = Dragon().represent(mol_list)

    # generate indices for splitting the features and targets into training and test sets
    train_indices, test_indices = train_test_split(range(10000), test_size=0.1, random_state=13)

    # initialize a ChemML MLP object
    tzvp_model = MLP()

    # load optimized hyperparameters into MLP object
    tzvp_model = load_hyperparameters(tzvp_model, 'dragon', 'polarizability', 'transfer_learning')

    # initialize a ChemML MLP object
    mlp = MLP()

    # load pre-trained ML model in the object
    from chemml.published.RI import load_model

    svp_model = mlp.load(load_model('dragon', 'polarizability'))

    # initialize a TransferLearning object
    from chemml.models import TransferLearning

    tl = TransferLearning(base_model=svp_model)

    # transfer the hidden layers from SVP model to TZVP model and fit the model to the new data
    combined_model = tl.transfer(dragon_tzvp[train_indices], targets[train_indices], tzvp_model)

    # predictions on test set
    y_pred = combined_model.predict(dragon_tzvp[test_indices])
    metrics_df = regression_metrics(targets[test_indices], y_pred)


Citation
+++++++++
Please cite the use of ChemML as:

::

    @article{Afzal2018a,
    author = {Afzal, Mohammad Atif Faiz and Cheng, Chong and Hachmann, Johannes},
    doi = {10.1063/1.5007873},
    journal = {The Journal of Chemical Physics},
    number = {24},
    pages = {241712},
    title = {{Combining first-principles and data modeling for the accurate prediction of the refractive index of organic polymers}},
    volume = {148},
    year = {2018}
    }

    @article{Afzal2019d,
    title={A deep neural network model for packing density predictions and its application in the study of 1.5 million organic molecules},
    author={Afzal, Mohammad Atif Faiz and Sonpal, Aditya and Haghighatlari, Mojtaba and Schultz, Andrew J and Hachmann, Johannes},
    journal={Chemical science},
    volume={10},
    number={36},
    pages={8374--8383},
    year={2019},
    publisher={Royal Society of Chemistry}
    }


    @article{Afzal2019a,
    author = {Afzal, Mohammad Atif Faiz and Hachmann, Johannes},
    doi = {10.1039/c8cp05492d},
    issn = {14639076},
    journal = {Physical Chemistry Chemical Physics},
    number = {8},
    pages = {4452--4460},
    publisher = {Royal Society of Chemistry},
    title = {{Benchmarking DFT approaches for the calculation of polarizability inputs for refractive index predictions in organic polymers}},
    volume = {21},
    year = {2019}
    }

    @article{Afzal2019b,
    author = {Afzal, Mohammad Atif Faiz and Haghighatlari, Mojtaba and {Prasad Ganesh}, Sai and Cheng, Chong and Hachmann, Johannes},
    issn = {1932-7447},
    journal = {The Journal of Physical Chemistry C},
    pages = {14610--14618},
    publisher = {American Chemical Society},
    title = {{ Accelerated Discovery of High-Refractive-Index Polyimides via First-Principles Molecular Modeling, Virtual High-Throughput Screening, and Data Mining }},
    volume = {123},
    year = {2019}
    }

    @article{vishwakarma2019towards,
    title={Towards autonomous machine learning in chemistry via evolutionary algorithms},
    author={Vishwakarma, Gaurav and Haghighatlari, Mojtaba and Hachmann, Johannes},
    journal={ChemRxiv preprint},
    year={2019}
    }

    @phdthesis{atif_thesis,
    author       = {Afzal, Mohammad Atif Faiz}, 
    title        = {From virtual high-throughput screening and machine learning to the discovery and rational design of polymers for optical applications},
    school       = {University at Buffalo},
    year         = {2018}
    }

    @mastersthesis{gaurav_msthesis,
    author = {Vishwakarma, Gaurav},
    school = {University at Buffalo},
    title = {{Machine Learning Model Selection for Predicting Properties of High-Refractive-Index Polymers}},
    year = {2018}
    }

    @article{vishwakarma2021metrics,
    title={Metrics for Benchmarking and Uncertainty Quantification: Quality, Applicability, and Best Practices for Machine Learning in Chemistry},
    author={Vishwakarma, Gaurav and Sonpal, Aditya and Hachmann, Johannes},
    journal={Trends in Chemistry},
    year={2021},
    publisher={Cell Press}
    }


    


    