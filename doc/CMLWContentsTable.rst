=============================
Table of Contents
=============================

This is a complete list of all the methods that the ChemML Wrapper is interfacing with.
The further details of each block, as the parameters and attributes of each block is provided in each function's page.

The table header provides information about:
    - task and subtask: just for an easier classification of methods
    - host: the main library/dependency required for using the method
    - function: the method name
    - input and output tokens: the only tokens that are responsible to send or receive data and information in each blocks

+----+--------------+-----------------+--------+-------------------+----------------------------------+----------------------------------------------------------+
|  # | task         | subtask         | host   | function          | input tokens                     | output tokens                                            |
+====+==============+=================+========+===================+==================================+==========================================================+
|  1 | Enter Data   | input           | pandas | :ref:`read_table` | no receiver                      | df                                                       |
+----+--------------+-----------------+--------+-------------------+----------------------------------+----------------------------------------------------------+
|  2 | Enter Data   | input           | pandas | :ref:`read_excel` | no receiver                      | df                                                       |
+----+--------------+-----------------+--------+-------------------+----------------------------------+----------------------------------------------------------+
|  3 | Prepare Data | basic operators | cheml  | :ref:`Split`      | df                               | df1, df2                                                 |
+----+--------------+-----------------+--------+-------------------+----------------------------------+----------------------------------------------------------+
|  4 | Prepare Data | basic operators | cheml  | :ref:`Merge`      | df1, df2                         | df                                                       |
+----+--------------+-----------------+--------+-------------------+----------------------------------+----------------------------------------------------------+
|  5 | Prepare Data | basic operators | cheml  | :ref:`PyScript`   | df1, df2, api1, api2, var1, var2 | df_out1, df_out2, api_out1, api_out2, var_out1, var_out2 |
+----+--------------+-----------------+--------+-------------------+----------------------------------+----------------------------------------------------------+



Data representation (features, descriptors, or latent variables) for chemical molecules and materials.

cheml
---------
+------------------------+------------------------------------------+-----------------------------------------+
| block                  | Inputs                                   | Outputs                                 |
+========================+==========================================+=========================================+
| :ref:`RDKFP`           | molfile                                  | df, removed_rows                        |
+------------------------+------------------------------------------+-----------------------------------------+
| :ref:`Dragon`          | molfile                                  | df                                      |
+------------------------+------------------------------------------+-----------------------------------------+
| :ref:`CoulombMatrix`   |                                          | df                                      |
+------------------------+------------------------------------------+-----------------------------------------+
| :ref:`BagofBonds`      |                                          | df                                      |
+------------------------+------------------------------------------+-----------------------------------------+
| :ref:`DistanceMatrix`  | df                                       | df                                      |
+------------------------+------------------------------------------+-----------------------------------------+


sklearn
---------

==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
PolynomialFeatures      df                  api, df
==================      ============        ============
----

----
=======================================================
Script
=======================================================

Python script block in the structure of computational graph

cheml
---------
+------------------------+------------------------------------------+----------------------------------------------------------+
| block                  | Inputs                                   | Outputs                                                  |
+========================+==========================================+==========================================================+
| :ref:`PyScript`        | df1, df2, api1, api2, var1, var2         | df_out1, df_out2, api_out1, api_out2, var_out1, var_out2 |
+------------------------+------------------------------------------+----------------------------------------------------------+
----

----
=======================================================
Input
=======================================================


Input blocks read input files and do basic changes in the data frames too.


| :ref:`Merge`           | df1, df2                                 | df                                      |
+------------------------+------------------------------------------+-----------------------------------------+
| :ref:`Split`           | df                                       | df1, df2                                |
+------------------------+------------------------------------------+-----------------------------------------+
----

----
=======================================================
Output
=======================================================

Output blocks are used for storing data frames and other type of output files.

cheml
---------
+------------------------+------------------------------------------+-----------------------------------------+
| block                  | Inputs                                   | Outputs                                 |
+========================+==========================================+=========================================+
| :ref:`SaveFile`        | df                                       | filepath                                |
+------------------------+------------------------------------------+-----------------------------------------+
----

----
=======================================================
Preprocessor
=======================================================

Preprocessor functions transfer raw feature vectors into a representation that is more suitable for the downstream estimators.

cheml
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
MissingValues           dfx, dfy            dfx, dfy, api
Trimmer                 dfx, dfy            dfx, dfy, api
Uniformer               dfx, dfy            dfx, dfy, api
Constant                df                  df, api, 'removed_columns_'
==================      ============        ============

sklearn
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
Imputer                 df                  api, df
StandardScaler          df                  api, df
MinMaxScaler            df                  api, df
MaxAbsScaler            df                  api, df
RobustScaler            df                  api, df
Normalizer              df                  api, df
Binarizer               df                  api, df
OneHotEncoder           df                  api, df
==================      ============        ============
----

----
=======================================================
Feature Transformation
=======================================================

Those dimension reduction methods that involve transformation of data to a new feature space.

cheml
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
TBFS                    df                  api, df
==================      ============        ============

sklearn
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
PCA                     df                  api, df
KernelPCA               df                  api, df
RandomizedPCA           df                  api, df
LDA                     df                  api, df
==================      ============        ============
----

----
=======================================================
Feature Selection
=======================================================

Those dimension reduction methods that determine which features should be used to address a particular problem based on their original values and not the transformed feature space.

cheml
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
TBFS                    df                  api, df
==================      ============        ============

sklearn
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
VarianceThreshold       df                  api, df
SelectKBest             df                  api, df
==================      ============        ============
----

----
=======================================================
Divider
=======================================================

To split data to smaller folds.

sklearn
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
Train_Test_Split        dfx, dfy            dfx_train, dfx_test, dfy_train, dfy_test
KFold                                       CV
==================      ============        ============
----

----
=======================================================
Regression
=======================================================

Regression methods.

cheml
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
TBFS                    df                  api, df
==================      ============        ============

sklearn
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
SVR                     dfx, dfy            r2_train, model
KernelRidge             dfx, dfy            r2_train, api
==================      ============        ============
----

----
=======================================================
Postprocessor
=======================================================

postprocessing tasks like evaluation, validation, model selection, ...

sklearn
---------
+------------------------+------------------------------------------+-----------------------------------------+
| block                  | Inputs                                   | Outputs                                 |
+========================+==========================================+=========================================+
| :ref:`GridSearchCV`    | dfx, dfy, estimator                      | \cv_results_, \best_estimator_, api     |
+------------------------+------------------------------------------+-----------------------------------------+
| Evaluation             | dfx, dfy, CV, X_scaler, Y_scaler, model  | results                                 |
+------------------------+------------------------------------------+-----------------------------------------+
----


----
=======================================================
Visualization
=======================================================

Regression methods.

matplotlib
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
TBFS                    df                  api, df
==================      ============        ============


seaborn
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
SVR                     dfx, dfy            r2_train, model
KernelRidge             dfx, dfy            r2_train, api
==================      ============        ============
----