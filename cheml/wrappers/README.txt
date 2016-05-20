## DataRepresentation
		module						function
		------						--------
		cheml						RDKitFingerprint
		cheml						Dragon
		cheml 						CoulombMatrix
		cheml 						BagofBonds
		sklearn						PolynomialFeatures

## Input
		module						function
		------						--------
		cheml						File  	:	read data
		cheml						Merge 	:	data - target -> data, target
		cheml 						Split  	:	data -> data, target

## Output
		module						function
		------						--------
		cheml						SaveFile
		cheml						settings	: path, error file, log file, pyscript, cheml script

## Preprocessor
		module						function
		------						--------
		cheml 						MissingValues
		cheml 						Trimmer
		cheml 						Uniformer
		sklearn						Imputer
		sklearn 					StandardScaler
		sklearn						MinMaxScaler
		sklearn						MaxAbsScaler
		sklearn						RobustScaler
		sklearn						Normalizer
		sklearn						Binarizer
		sklearn						OneHotEncoder

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

