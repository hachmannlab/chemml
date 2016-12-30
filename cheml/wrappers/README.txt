## DataRepresentation

    module						function            legal_inputs                        legal_outputs
    ------						--------            ------------                        -------------
D	cheml						RDKitFingerprint    {molfile}                           {df}
D	cheml						Dragon              {molfile}                           {df}
    cheml 						CoulombMatrix       {}                                  {df}
    cheml 						BagofBonds          {}                                  {df}
    cheml                       DistanceMatrix      {df}                                {df}
    sklearn						PolynomialFeatures  {df}                                {api,df}


----------------------------------------------------------------------------------------------------
## Script

D   cheml                       PyScript            {df,api,value}                            {df,api,value}


----------------------------------------------------------------------------------------------------
## Input

    module						function            legal_inputs                        legal_outputs
    ------						--------            ------------                        -------------
D	cheml						File  	            {}                                  {df}
D	cheml						Merge               {df1,df2}                           {df}
D	cheml 						Split  		        {df}                                {df1,df2}


----------------------------------------------------------------------------------------------------
## Output

    module						function
    ------						--------
D	cheml						SaveFile            {df}                         {filepath}                                {fp(file_path)}


----------------------------------------------------------------------------------------------------
## Preprocessor

    module						function            legal_inputs                        legal_outputs
    ------						--------            ------------                        -------------
D   cheml 						MissingValues       {df}                                {api,df}
D   cheml 						Trimmer             {dfx, dfy}                          {api,dfx,dfy}
D   cheml 						Uniformer           {dfx, dfy}                          {api,dfx,dfy}
D	sklearn						Imputer             {df}                                {api,df}
D	sklearn 					StandardScaler      {df}                                {api,df}
D	sklearn						MinMaxScaler        {df}                                {api,df}
D	sklearn						MaxAbsScaler        {df}                                {api,df}
D	sklearn						RobustScaler        {df}                                {api,df}
D	sklearn						Normalizer          {df}                                {api,df}
D	sklearn						Binarizer           {df}                                {api,df}
D	sklearn						OneHotEncoder       {df}                                {api,df}


----------------------------------------------------------------------------------------------------
## FeatureSelection

    module						function
    ------						--------
    cheml						TBFS
    sklearn						VarianceThreshold
    sklearn						SelectKBest
    sklearn						SelectPercentile
    sklearn						SelectFpr
    sklearn						SelectFdr
    sklearn						SelectFwe
    sklearn						RFE
    sklearn						RFECV
    sklearn						SelectFromModel


----------------------------------------------------------------------------------------------------
## FeatureTransformation

    module                      function            legal_inputs                        legal_outputs
    ------                      --------            ------------                        -------------
D   sklearn                     PCA                 {df}                                {api,df}
    sklearn                     KernelPCA
    sklearn                     RandomizedPCA
    sklearn                     LDA

----------------------------------------------------------------------------------------------------
## Divider

    module						function            legal_inputs                        legal_outputs
    ------						--------            ------------                        -------------
D   sklearn						Train_Test_Split    {dfx,dfy(None)}                     {dfx_train,dfx_test,dfy_train(None),dfy_test(None)}
D   sklearn						KFold               {}                                  {CV}
    sklearn						StratifiedKFold
    sklearn						LabelKFold
    sklearn						LeaveOneOut
    sklearn						LeavePOut
    sklearn						LeaveOneLabelOut
    sklearn						LeavePLabelOut
    sklearn						ShuffleSplit
    sklearn						LabelShuffleSplit
    sklearn						PredefinedSplit


----------------------------------------------------------------------------------------------------
## Regression

    module						function            legal_inputs                        legal_outputs
    ------						--------            ------------                        -------------
    cheml						NN_MLP_PSGD
    cheml						NN_MLP_DSGD
    cheml						NN_MLP_Theano
    cheml						NN_MLP_Tensorflow
    cheml						SVR
D   sklearn						Linear
D   sklearn						Ridge
D   sklearn						KernelRidge
D   sklearn						Lasso
D   sklearn                     MultiTaskLasso
D   sklearn                     ElasticNet
D   sklearn                     MultiTaskElasticNet
D   aklearn                     Lars
D   sklearn						LassoLars
D   sklearn                     BayesianRidge
D   sklearn                     ARD
D   sklearn                     Logistic
D   sklearn                     SGD
D   sklearn						SVR                 {'dfx','dfy','pack'}          {'r2_train','api'}
D   sklearn						NuSVR
D   sklearn						LinearSVR


----------------------------------------------------------------------------------------------------
## Classification

    module						function
    ------						--------


----------------------------------------------------------------------------------------------------
## Postprocessor

    module						function
    ------						--------
D   sklearn                     Evaluation
D   sklearn                     Grid_SearchCV
    sklearn                     Model_Persistence




----------------------------------------------------------------------------------------------------
## Visualization

    module						function
    ------						--------
    cheml


----------------------------------------------------------------------------------------------------
## Optimizer

    module						function
    ------						--------
    cheml						GA_Binary
    cheml						GA_Real
    cheml						ANT
    cheml						PSO

