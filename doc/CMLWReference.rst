=============================
ChemML Wrapper reference
=============================

This is the reference of the computational graph blocks provided by ChemML Wrapper. The further details of each block, as the
parameters and attributes of each block is provided in the :ref:`genindex`.

----
=======================================================
Data Representation
=======================================================

Data representation (features, descriptors, latent variables) for chemical molecules and materials.

cheml
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
RDKitFingerprint        molfile             df
Dragon                  molfile             df
CoulombMatrix                               df
BagofBonds                                  df
DistanceMatrix          df                  df
==================      ============        ============

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
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
PyScript                df,api,value        df,api,value
==================      ============        ============
----

----
=======================================================
Input
=======================================================


Input blocks deal with all sort of input files and data frames.

cheml
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
File                                        df
Merge                   df1, df2            df
Split                   df                  df1, df2
==================      ============        ============
----

----
=======================================================
Output
=======================================================

Output blocks are used for storing data frames and other type of output files.

cheml
---------
==================      ============        ============
block                   Inputs              Outputs
==================      ============        ============
SaveFile                df                  filepath
==================      ============        ============
