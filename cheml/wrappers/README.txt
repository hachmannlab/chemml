## DataRepresentation
		module						function            legal_inputs                        legal_outputs
		------						--------            ------------                        -------------
		cheml						RDKitFingerprint    {}                                  {df}
		cheml						Dragon              {}                                  {df}
		cheml 						CoulombMatrix       {}                                  {df}
		cheml 						BagofBonds          {}                                  {df}
		cheml                       DistanceMatrix      {df}                                {df}
		sklearn						PolynomialFeatures  {df}                                {api,df}

## Input
		module						function            legal_inputs                        legal_outputs
		------						--------            ------------                        -------------
		cheml						File  	            {}                                  {df}
		cheml						Merge               {df1,df2}                           {df}
		cheml 						Split  		        {df}                                {df1,df2}

## Output
		module						function
		------						--------
		cheml						SaveFile            {df}                                {api,df}
		cheml						settings

## Preprocessor
		module						function            legal_inputs                        legal_outputs
		------						--------            ------------                        -------------
		cheml 						MissingValues
		cheml 						Trimmer
		cheml 						Uniformer
		sklearn						Imputer             {df}                                {api,df}
		sklearn 					StandardScaler      {df}                                {api,df}
		sklearn						MinMaxScaler        {df}                                {api,df}
		sklearn						MaxAbsScaler        {df}                                {api,df}
		sklearn						RobustScaler        {df}                                {api,df}
		sklearn						Normalizer          {df}                                {api,df}
		sklearn						Binarizer           {df}                                {api,df}
		sklearn						OneHotEncoder       {df}                                {api,df}

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

## FeatureTransformation
		module						function
		------						--------
		sklearn						PCA
		sklearn						KernelPCA
		sklearn						RandomizedPCA
		sklearn						LDA

## Divider
		module						function
		------						--------
		sklearn						train_test_split
		sklearn						K-fold
		sklearn						StratifiedKFold
		sklearn						LabelKFold
		sklearn						LeaveOneOut
		sklearn						LeavePOut
		sklearn						LeaveOneLabelOut
		sklearn						LeavePLabelOut
		sklearn						ShuffleSplit
		sklearn						LabelShuffleSplit
		sklearn						PredefinedSplit

## Regression
		module						function
		------						--------
		cheml						NN_MLP_PSGD
		cheml						NN_MLP_DSGD
		cheml						NN_MLP_Theano
		cheml						NN_MLP_Tensorflow
		cheml						SVR
		sklearn						Linear
		sklearn						Ridge
		sklearn						KernelRidge
		sklearn						Lasso
		sklearn						ElasticNet
		sklearn						LassoLars
		sklearn						SVR
		sklearn						NuSVR
		sklearn						LinearSVR

## Classification
		module						function
		------						--------

## Evaluation
		module						function
		------						--------
		sklearn						r2_score
		sklearn						mean_absolute_error
		sklearn						median_absolute_error
		sklearn						mean_squared_error
		sklearn						root_mean_squared_error
		sklearn						explained_variance_score

## Visualization
		module						function
		------						--------
		cheml

## Optimizer
		module						function
		------						--------
		cheml						GA_Binary
		cheml						GA_Real
		cheml						ANT
		cheml						PSO

