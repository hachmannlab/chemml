#!/usr/bin/env python

PROGRAM_NAME = "CheML"
PROGRAM_VERSION = "v0.0.1"
REVISION_DATE = "2015-06-23"
AUTHORS = "Johannes Hachmann (hachmann@buffalo.edu) and Mojtaba Haghighatlari (mojtabah@buffalo.edu)"
CONTRIBUTORS = """ """
DESCRIPTION = "CheML is a machine learning and informatics program suite for the chemical and materials sciences."

# Version history timeline (move to CHANGES periodically):
# v0.0.1 (2015-06-02): complete refactoring of original CheML code in new package format


###################################################################################################
#TODO:
# -restructure more general functions into modules
###################################################################################################

import sys
import os
import shutil
import glob
import time
import argparse
import copy
import datetime

import numpy as np
import pandas as pd
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#1 remember to check if label file has a header or not
#2 

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

 									# FUNCTIONS #		
 									#   CheML   #	

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*									  

def std_datetime_str(mode='datetime'):
    """(std_time_str):
        This function gives out the formatted time as a standard string, i.e., YYYY-MM-DD hh:mm:ss.
    """
    if mode == 'datetime':
        return str(datetime.datetime.now())[:19]
    elif mode == 'date':
        return str(datetime.datetime.now())[:10]
    elif mode == 'time':
        return str(datetime.datetime.now())[11:19]
    elif mode == 'datetime_ms':
        return str(datetime.datetime.now())
    elif mode == 'time_ms':
        return str(datetime.datetime.now())[11:]
    else:
        sys.exit("Invalid mode!")

##################################################################################################

def tot_exec_time_str(time_start):
    """(tot_exec_time_str):
        This function gives out the formatted time string.
    """
    time_end = time.time()
    exec_time = time_end-time_start
    tmp_str = "Total execution time: %0.2fs (%dh %dm %0.2fs)" %(exec_time, exec_time/3600, (exec_time%3600)/60,(exec_time%3600)%60)
    return tmp_str

##################################################################################################
    
def banner(logfile, PROGRAM_NAME, PROGRAM_VERSION, REVISION_DATE, AUTHORS, DESCRIPTION,):
    """(banner):
        Banner for this script.
    """
    str = []
    str.append("============================================================================== ")
    str.append(SCRIPT_NAME + " " + SCRIPT_VERSION + " (" + REVISION_DATE + ")")
    str.append(AUTHOR)
    str.append("============================================================================== ")
    str.append(time.ctime())
    str.append("")    
    str.append(DESCRIPTION)
    str.append("")

    print 
    for line in str:
        print line
        logfile.write(line + '\n')

##################################################################################################
        
def print_function_opts(fline,logfile):
	"""(print_function_opts):
        Prints the invoked options to stdout and the logfile.
    """ 
	tmp_str = "Invoked script line: "
	print tmp_str
	logfile.write(tmp_str + '\n')  
	print fline
	print 
	logfile.write(fline + '\n\n')
	
	tmp_str = "============================================================================== "
	print
	print tmp_str
	logfile.write('\n' + tmp_str + '\n')
	
###################################################################################################

def initialization(argument):
	"""(initialization):
		Read initial files for the grade_master script.
	"""
	
	## default options
	data_path = None; label_path = None; log_file = "log.txt";  error_file = "error.txt"; df = 1 ; cn = 2 
	list_args=["data_path","label_path","log_file","error_file","df","cn"]
	
	## read options
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in initialization function."
			sys.exit(tmp_str)  
	
			
	# program files
	logfile = open(output_folder+'/'+log_file,'a',0)
	errorfile = open(output_folder+'/'+error_file,'a',0)
    
	banner(logfile, PROGRAM_NAME, PROGRAM_VERSION, REVISION_DATE, AUTHORS, DESCRIPTION)
	
	# check that file exists
	if data_path == None or label_path == None:
		tmp_str = "... data or label file not specified!"
		print tmp_str
		logfile.write(tmp_str + '\n')
		errorfile.write(tmp_str + '\n')

		tmp_str = "Aborting due to missing data or label file!"
		logfile.write(tmp_str + '\n')
		errorfile.write(tmp_str + '\n')
		sys.exit(tmp_str) 
	
	tmp_str = "=============================================================================="
	print tmp_str
	logfile.write(tmp_str + '\n\n')
	
	# give out options of this function
	fline = "funnction: initialization\n   data_path = %s\n   label_path = %s\n   log_file = %s\n   error_file = %s\n   df = %s\n   cn = %s\n" %(data_path,label_path,log_file,error_file,df,cn)	
	print_function_opts(fline,logfile)
	
	
	# make a logfile entry and screen entry so that we know where we stand
	tmp_str = "Starting data acquisition..."	
	print tmp_str
	logfile.write(tmp_str + '\n')
	
	# read input files
	data = pd.read_csv(data_path, skiprows=1,header=None)
	label = pd.read_csv(label_path, header=None)
	
	""" trim original data file """
	## find indices based on sorted labels 
	in_file = label
	in_file[1] = in_file.index
	in_file = in_file.sort(columns=0)
	
	## cut off number
	numel_h = pd.DataFrame(in_file[0].value_counts())
	numel_h = numel_h[numel_h[0]<cn]
	remove_list = numel_h.index
	for i in remove_list :
		in_file = in_file[in_file[0] != i]	
	in_file.index = pd.Index(range(len(in_file)))
	
	## shuffle in_file
	in_file = in_file.reindex(np.random.permutation(in_file.index))
	in_file.index = pd.Index(range(len(in_file)))
	
	## what fraction of dataset should be selected?
	nd = len(in_file)	# number of datapoints
	ns = math.floor(df * nd)	# number of samples
	sample_DS = random.sample(range(0,nd), int(ns))
	in_file = in_file.iloc[sample_DS,:]
	in_file.index = pd.Index(range(len(in_file)))
	labels = pd.DataFrame(in_file[0])
	
	## find sample points in the original data 
	data = data.iloc[list(in_file[1]),:]
	data.index = pd.Index(range(len(data)))
	
	tmp_str = "   Number of samples : %s" %len(data)
	print tmp_str
	logfile.write(tmp_str + '\n')
	
	tmp_str = "   Number of features : %s" %len(data.columns)
	print tmp_str
	logfile.write(tmp_str + '\n')
	
	tmp_str = "...data acquisition finished."
	print tmp_str
	logfile.write(tmp_str + '\n')
		
	tmp_str = "=============================================================================="
	print
	print tmp_str
	logfile.write('\n' + tmp_str + '\n\n')


	return data, labels, logfile, errorfile

###################################################################################################

def feature_scaled(ds1, ds2):
	""" (feature scaling):
		normalize TrS and TeS 
	"""
	pd.options.mode.chained_assignment = None
	for feature in ds1.columns:
		avg = np.mean(ds1[feature])
		std = np.std(ds1[feature])
	
		ds1[feature] = (ds1[feature]-avg)/std 
		ds2[feature] = (ds2[feature]-avg)/std

	return ds1, ds2

def feature_scaled_single(data):
	""" (feature scaling one input file):
		normalize TrS and TeS 
	"""
	pd.options.mode.chained_assignment = None
	ds1 = copy.deepcopy(data)
	for feature in ds1.columns:
		avg = np.mean(ds1[feature])
		std = np.std(ds1[feature])
		ds1[feature] = (ds1[feature]-avg)/std

	return ds1

###################################################################################################

def TrS_TeS(data, labels, arg_trf, arg_scaled):
	""" 
		make training set(TrS) and test set(TeS) 
	"""
	import random
	import math
	
	# what fraction of dataset should be training set?
	nd = len(data)	# number of datapoints	
	sample_TrS=random.sample(range(0,nd), int(math.floor(arg_trf*nd)))
	sample_TeS=list(set(range(0,nd)).difference(sample_TrS))
	
	TrS_data = data.iloc[sample_TrS,:]
	TrS_label = labels.iloc[sample_TrS,:] 
	TrS_data.index = pd.Index(range(len(TrS_data)))
	TrS_label.index = pd.Index(range(len(TrS_label)))
	
	TeS_data = data.iloc[sample_TeS,:]
	TeS_label = labels.iloc[sample_TeS,:]
	TeS_data.index = pd.Index(range(len(TeS_data)))
	TeS_label.index = pd.Index(range(len(TeS_label)))
	
	if arg_scaled == True:
		TrS_data, TeS_data = feature_scaled(TrS_data, TeS_data)
	
	return TrS_data, TrS_label, TeS_data, TeS_label

###################################################################################################
 
def CV(clf, data, labels, arg_trf, arg_scaled, n_folds):
	""" 
		make training set(TrS) and test set(TeS) for KFold cross validation by scikitlearn
	"""
	from sklearn.cross_validation import KFold
	import math
	
	nd = len(data)	# number of datapoints
	kf = KFold(nd, n_folds = n_folds, shuffle = True, random_state = None)
	
	MAE = []
	MAE_SD = []
	RMSE = []
	RMSE_SD = []
		
	for train_index, test_index in kf:
		# training set
		TrS_data = data.iloc[train_index,:]
		TrS_label = labels.iloc[train_index,:] 
		TrS_data.index = pd.Index(range(len(TrS_data)))
		TrS_label.index = pd.Index(range(len(TrS_label)))
		# test set
		TeS_data = data.iloc[test_index,:]
		TeS_label = labels.iloc[test_index,:]
		TeS_data.index = pd.Index(range(len(TeS_data)))
		TeS_label.index = pd.Index(range(len(TeS_label)))
		
		if arg_scaled == True:
			TrS_data, TeS_data = feature_scaled(TrS_data, TeS_data)

		clf.fit(TrS_data,TrS_label)
		clf.predict(TeS_data)
		
		MAE.append()
		MAE_SD.append()
		RMSE.append()
		RMSE_SD.append()
		
	
		
	return MAE, MAE_SD, RMSE, RMSE_SD

###################################################################################################
 
def pca(data,labels, logfile, errorfile,argument):
	""" Principal Component Analysis 
		by Scikitlearn 
	"""
	
	## default argument values 
	n_components=None; copy=True; whiten=False; evr_=False 
	list_args=["n_components","copy","whiten","evr_"]
	
	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in pca function."
			sys.exit(tmp_str)  
	
	# give out options of this function
	fline = "funnction: pca\n   n_components = %s\n   copy = %s\n   whiten = %s\n   evr_ = %s\n" %(n_components,copy,whiten,evr_)	
	print_function_opts(fline,logfile)
	
	from sklearn.decomposition import PCA
	pca = PCA(n_components=n_components,copy=copy, whiten=whiten)
	data = data.values
	data = pca.fit_transform(data)
	data = pd.DataFrame(data)	
	
	if evr_ == True:
		eigenvalue = pca.explained_variance_ratio_
		eigenvalue = pd.DataFrame({0:list(eigenvalue)})
		eigenvalue.to_csv(output_folder+'/pca_eigenvalues_%s.csv'%time.ctime())
	
	return data, labels, logfile, errorfile
		
###################################################################################################

def kpca(data,labels, logfile, errorfile,argument):
	""" 
		KernelPCA from Scikitlearn 
	""" 
	## default argument values
	n_components=None; kernel="linear";gamma=None; degree=3; coef0=1; kernel_params=None;alpha=1.0; fit_inverse_transform=False; eigen_solver='auto';tol=0; max_iter=None; remove_zero_eig=False; lambdas_ = False
	list_args=["n_components","kernel","gamma","degree","coef0","kernel_params","alpha","fit_inverse_transform","eigen_solver","tol","max_iter","remove_zero_eig","lambdas_"]
	
	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in kpca function."
			sys.exit(tmp_str)  
	
	# give out options of this function
	fline = "funnction: kpca\n   n_components = %s\n   kernel = %s\n   gamma = %s\n   degree = %s\n   coef0 = %s   kernel_params = %s\n   alpha = %s\n   fit_inverse_transform = %s\n   eigen_solver = %s\n   tol = %s\n   max_iter = %s\n   remove_zero_eig = %s\n   lambdas_ = %s\n" %(n_components,kernel,gamma,degree,coef0,kernel_params,alpha,fit_inverse_transform,eigen_solver,tol,max_iter,remove_zero_eig,lambdas_)	
	print_function_opts(fline,logfile)
	
	from sklearn.decomposition import KernelPCA
	kpca = KernelPCA(n_components=n_components, kernel = kernel , degree = degree , gamma = gamma , coef0 = coef0 , alpha = alpha , kernel_params = kernel_params , fit_inverse_transform = fit_inverse_transform , eigen_solver = eigen_solver , tol = tol , max_iter = max_iter , remove_zero_eig = remove_zero_eig )
	data = data.values
	data = kpca.fit(data).transform(data)
	data = pd.DataFrame(data)
	
	if lambdas_ == True:
		eigenvalue = kpca.lambdas_
		eigenvalue = pd.DataFrame({0:list(eigenvalue)})
		eigenvalue.to_csv(output_folder+'/kpca_eigenvalues_%s.csv'%time.ctime())
			
	return data, labels	, logfile, errorfile	 

###################################################################################################

def lda(data,labels, logfile, errorfile,argument):
	""" 
		LDA from Scikitlearn 
	""" 

	## default argument values 
	solver="svd"; shrinkage=None; priors=None;n_components=None; store_covariance=False; tol=1e-4 
	trf=1.0; scaled = True 
	list_args=["solver","shrinkage","priors","n_components","store_covariance","tol","trf","scaled"]
	
	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in lda function."
	
	# give out options of this function
	fline = "funnction: lda\n   solver = %s\n   shrinkage = %s\n   priors = %s\n   n_components = %s\n   store_covariance = %s\n   tol = %s\n   trf = %s\n   scaled = %s\n" %(solver,shrinkage,priors,n_components,store_covariance,tol,trf,scaled)
	print_function_opts(fline,logfile)
	
	## TrS/TeS and normalization
	Data={}
	Labels={}
	if trf < 1 and trf>0:
		Data[0], Labels[0], Data[1], Labels[1] = TrS_TeS(data, labels, trf, scaled)
	elif trf ==1 or trf==0:
		Labels[0] = labels
		if scaled ==True:
			Data[0] = feature_scaled_single(data)
		else:
			Data[0] = copy.deepcopy(data)
			
	## LDA
	from sklearn.lda import LDA
	lda = LDA(n_components=n_components) #solver=solver, shrinkage=shrinkage, priors=priors,store_covariance=store_covariance, tol=tol
	X,y = Data[0].values, Labels[0][0].values
	lda.fit(X,y)
	#y=y/.01
	#r2_train = lda.fit(X,y).score(X,y)
	
	moment = str(time.ctime())
	with open(output_folder+'/LDA_results_%s.txt'%moment,'a') as file:
		file.write('                    Linear Discriminant Analysis Results\n')
		file.write('==============================================================================\n')
		file.write('\n')
		file.write('----------------------\n')
		file.write(' Training Set Results \n')
		file.write('----------------------\n')
		file.write('\n')
		file.write('Number of samples : '+str(len(X))+'\n')	
		file.write('Number of features : '+str(len(data.columns))+'\n')
		#file.write('coefficient of determination R^2: '+str(r2_train)+'\n')
		file.write('==============================================================================\n')
	
	if len(Data)>1:
		Xt = Data[1].values
		#yt = Labels[1][0].values
		#yt=yt/0.01
		#r2_test = lda.score(Xt,yt)
		data = lda.transform(Xt)
		data = pd.DataFrame(data)	
		labels = Labels[1]
		
		with open(output_folder+'/LDA_results_%s.txt'%moment,'a') as file:
			file.write(' \n')
			file.write('------------------\n')
			file.write(' Test Set Results \n')
			file.write('------------------\n')
			file.write(' \n')
			file.write('Number of samples : '+str(len(data))+'\n')	
			file.write('Number of features : '+str(len(data.columns))+'\n')
			#file.write('coefficient of determination, R^2: '+str(r2_test)+'\n')			
			file.write('==============================================================================\n')
	else:		
		data = lda.transform(X)
		data = pd.DataFrame(data)	
		labels = Labels[0]
		with open(output_folder+'/LDA_results_%s.txt'%moment,'a') as file:
			file.write(' \n')
			file.write('All the data points have been transformed (no test set)!  \n')
			file.write(' \n')
			file.write('==============================================================================\n')
			file.write('                                     END\n')
			
	return data, labels, logfile, errorfile

###################################################################################################

def svr(data,labels, logfile, errorfile,argument):
	""" 
		Support Vector Regression from Scikitlearn 
	"""
	
	## default argument values 
	kernel='rbf'; degree=3; gamma=0.0; coef0=0.0; tol=1e-3; C=1.0; epsilon=0.1; shrinking=True; cache_size=200; verbose=False; max_iter=-1
	trf = 1.0; scaled = True
	list_args=["kernel","gamma","degree","coef0","tol","C","epsilon","shrinking","cache_size","verbose","max_iter","trf","scaled"]

	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in svr function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: svr\n   kernel = %s\n   gamma = %s\n   degree = %s\n   coef0 = %s\n   tol = %s\n   C = %s\n   epsilon = %s\n   shrinking = %s\n   cache_size = %s\n   verbose = %s\n   max_iter = %s\n   trf = %s\n   scaled = %s\n" %(kernel,gamma,degree,coef0,tol,C,epsilon,shrinking,cache_size,verbose,max_iter,trf,scaled)	
	print_function_opts(fline,logfile)
	
	## TrS/TeS and normalization
	Data={}
	Labels={}
	if trf < 1 and trf>0:
		Data[0], Labels[0], Data[1], Labels[1] = TrS_TeS(data, labels, trf, scaled)
	elif trf ==1 or trf==0:
		Labels[0] = labels
		if scaled ==True:
			Data[0] = feature_scaled_single(data)
		else:
			Data[0] = copy.deepcopy(data)
			
	## SVR
	from sklearn.svm import SVR
	svr = SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)
	
	X,y = Data[0].values, Labels[0][0].values
	r2_training = svr.fit(X,y).score(X,y)
	
	moment = str(time.ctime())
	with open(output_folder+'/SVR_results_%s.txt'%moment,'a') as file:
		file.write('                      Support Vector Regression Results\n')
		file.write('==============================================================================\n')
		file.write('\n')
		file.write('----------------------\n')
		file.write(' Training Set Results \n')
		file.write('----------------------\n')
		file.write('\n')
		file.write('Number of samples : '+str(len(X))+'\n')	
		file.write('Number of features : '+str(len(data.columns))+'\n')
		file.write('coefficient of determination R^2: '+str(r2_training)+'\n')
		file.write('==============================================================================\n')
	
	if len(Data)>1:
		Xt = Data[1].values
		yt = Labels[1][0].values
		r2_test = svr.score(Xt,yt)
		yt_predict = svr.predict(Xt)
		dif=abs(yt_predict-yt)
		MAE=np.mean(dif)
		MAE_std = np.std(dif)
		MSE=np.mean(dif**2)
		RMSE=np.sqrt(MSE)
		RMSE_std=np.sqrt(np.std(dif**2))
	
		with open(output_folder+'/SVR_results_%s.txt'%moment,'a') as file:
			file.write(' \n')
			file.write('------------------\n')
			file.write(' Test Set Results \n')
			file.write('------------------\n')
			file.write(' \n')
			file.write('coefficient of determination, R^2: '+str(r2_test)+'\n')			
			file.write('Mean Absolute Error +/- Standard Deviation: '+str(MAE)+' +/- '+str(MAE_std)+'\n')						
			file.write('Root Mean Squared Error +/- Standard Deviation: '+str(RMSE)+' +/- '+str(RMSE_std)+'\n')						
			file.write('==============================================================================\n')
			file.write('                                     END\n')
			
	return data,labels, logfile, errorfile
 		
###################################################################################################

def pre_identical(data,labels, logfile, errorfile,argument):
	""" 
		remove features with many identical values 
	"""
	## default argument values 
	f_remove = 10
	list_args=["f_remove"]
	
	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in pre_identical function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: pre_identical\n   f_remove = %s\n" %(f_remove)	
	print_function_opts(fline,logfile)
	
	## check identical values
	count = 0
	max_size_class = list(labels[0].value_counts())[0]
	for i in data.columns:
		max_num_des = list(data[i].value_counts())[0]
		if max_num_des > f_remove * max_size_class:
			data = data.drop(i,1)
			count+=1
	
	print "Number of removed features with 'pre_identical' function: ", count			
		
	return data,labels,logfile, errorfile

###################################################################################################

def pre_linearDep(data,labels, logfile, errorfile,argument):
	""" 
		remove features with linear dependency 
	"""

	## default argument values 
	min_r2 = 0.99 ; n_iter = 20
	onebyone = True ; rank = False 
	list_args=["min_r2","n_iter","onebyone","rank"]
	
	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in pre_linearDep function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: pre_linearDep\n   min_r2 = %s\n   n_iter = %s\n   onebyone = %s\n   rank = %s\n" %(min_r2,n_iter,onebyone,rank)	
	print_function_opts(fline,logfile)
	
	## import modules
	from numpy.linalg import matrix_rank
	from sklearn import linear_model
	import random
	
	## step 1
	if onebyone == True:
		remove_list = []
		n = len(data.columns)
		col = data.columns
		for i in range(0,n):
			for j in range(i+1,n):
				X = np.reshape(data[col[i]].values , (-1, 1))
				y = np.reshape(data[col[j]].values , (-1, 1))
				regr = linear_model.LinearRegression()
				regr.fit(X,y)
				if regr.score(X,y) >= min_r2 : 
					remove_list.append(col[j])

		remove_list = np.unique(remove_list)
		print "Number of removed dependant linear features with 'onebyone' checking: ", len(remove_list)			
		data = data.drop(remove_list,1)								
	
	## step 2
	if rank == True:
		k = len(data.columns) - matrix_rank(data.values)
		if k > 0:
			print 'Number of linearly dependent features: ', k
			
			lin_candidates = []
			for i in data.columns:
				df = data.drop(i,1)
				if len(data.columns) - matrix_rank(df.values) - k >= 1 :
					lin_candidates.append(i)
			
			a = []
			b = len(data.columns) - k
			Bool = 0		
			for i in range(0,n_iter):
				remove_list = random.sample(lin_candidates,k)			
				df = data.drop(remove_list,1)
				rankdf = matrix_rank(df.values)
				if rankdf + k == len(data.columns):									
					a = remove_list
					Bool = 1
					break
				elif rankdf + k > b:
					b = rankdf + k 
					a = remove_list	
					Bool = 2
					
			if Bool == 0:
				print "SVD rank method didn't remove any of features"
			if Bool == 1:
				print "SVD rank method removed all the linearly dependant features"			
			if Bool == 2:
				print "SVD rank method removed some of the linearly dependant features. Some independent features may have been removed as well."			
			
			data = data.drop(a,1)		
		
		else:
			print "No linearly dependent features have been determined by SVD rank method"	
	
	return data,labels, logfile, errorfile	

###################################################################################################	

def ols(data,labels, logfile, errorfile,argument):
	""" 
		Ordinary Least Squares from statsmodel
	"""
	## default argument values 
	trf = 1.0; scaled = True
	list_args=["trf","scaled"]
	
	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in ols function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: ols\n   trf = %s\n   scaled = %s\n" %(trf,scaled)	
	print_function_opts(fline,logfile)
	
	## TrS/TeS and normalization
	Data={}
	Labels={}
	if trf < 1 and trf>0:
		Data[0], Labels[0], Data[1], Labels[1] = TrS_TeS(data, labels, trf, scaled)
	elif trf ==1 or trf==0:
		Labels[0] = labels
		if scaled ==True:
			Data[0] = feature_scaled_single(data)
		else:
			Data[0] = copy.deepcopy(data)
			
	## OLS
	import statsmodels.api as sm
	
	X = Data[0]
	X['c'] = [1]*len(X)
	X = X.values
	y = Labels[0][0].values
	
	model = sm.OLS(y, X)
	results = model.fit()
	
	moment = str(time.ctime())
	with open(output_folder+'/OLS_results_%s.txt'%moment,'a') as file:
		file.write('\n')
		file.write('----------------------\n')
		file.write(' Training Set Results \n')
		file.write('----------------------\n')
		file.write(str(results.summary()))
	
	param = results.params
	
	if len(Data)>1:
		Xt = Data[1]
		Xt['c'] = [1]*len(Xt)
		Xt = Xt.values
		yt = Labels[1][0].values
		
		myt= param * Xt			# n*m
		predicted_yt = np.array([sum(i) for i in myt])
		
		AE = np.abs(predicted_yt - yt)
		MAE = np.mean(AE)
		MAE_std = np.std(AE)
		MSE = np.mean(AE**2)
		RMSE = np.sqrt(MSE)
		RMSE_std = np.sqrt(np.std(AE**2))
		
		dfAE = pd.DataFrame(AE, columns=['Absolute Error'])
		features_index=['x'+str(i) for i in data.columns]		
		
		with open(output_folder+'/OLS_results_%s.txt'%moment,'a') as file:
			file.write('\n')
			file.write('\n')
			file.write('Input Features:'+str(features_index) + '\n')
			file.write('==============================================================================\n')
			file.write(' \n')
			file.write(' \n')
			file.write('------------------\n')
			file.write(' Test Set Results \n')
			file.write('------------------\n')
			dfAE.describe().to_csv(file,sep='\t')	
			file.write(' \n')
			file.write('Mean Absolute Error +/- Standard Deviation : ' + str(MAE)+' +/- '+str(MAE_std)+'\n')						
			file.write('Root Mean Squared Error +/- Standard Deviation : ' + str(RMSE)+' +/- '+str(RMSE_std)+'\n')						
			file.write('==============================================================================\n')
			file.write('                                     END\n')
			
	return data,labels, logfile, errorfile

###################################################################################################

def mvlr(data,labels, logfile, errorfile,argument):
	""" 
		Multivariate Linear Regression by Gradient Descent
	"""
	## default argument values 
	trf = 1.0 ; scaled = True ; MaxIt = 200 ; alpha = 1.0/len(data.columns) ; save_every = int(MaxIt/2)   
	list_args=["trf","scaled","MaxIt","alpha","save_every"]
	
	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in mvlr function."
			sys.exit(tmp_str)
					
	## give out options of this function
	fline = "funnction: mvlr\n   trf = %s\n   scaled = %s\n   MaxIt = %s\n   alpha = %s\n   save_every = %s\n" %(trf,scaled,MaxIt,alpha,save_every)	
	print_function_opts(fline,logfile)
	
	## TrS/TeS and normalization
	Data={}
	Labels={}
	if trf < 1 and trf>0:
		Data[0], Labels[0], Data[1], Labels[1] = TrS_TeS(data, labels, trf, scaled)
	elif trf ==1 or trf==0:
		Labels[0] = labels
		if scaled ==True:
			Data[0] = feature_scaled_single(data)
		else:
			Data[0] = copy.deepcopy(data)
	
	
	## Initialization
		# Problem Definition: h(x)=(t1 * x1) + (t2 * x2) + ... + (tn * xn)) + (t0*x0)  >>> Note:  x0=1 
	
	save_file=[i*save_every for i in range(0,MaxIt/save_every+1)]
	thetas=[0.5]*(len(data.columns)+1) 			# is a dictionary of data file to have all the features as variables
	
	X = Data[0]
	X['const'] = [1]*len(X)
	Y = Labels[0]
	nData = len(X)
	
	moment = str(time.ctime())
	with open(output_folder+'/MVLR_results_%s.txt'%moment,'a') as file:
		file.write('\n')
		file.write('                        MultiVariate Linear Regression\n')
		file.write('==============================================================================\n')
		file.write('----------------------\n')
		file.write(' Training Set Results \n')
		file.write('----------------------\n')

	## Training, GradientDescent 	# Problem Definition: tj = tj - (alpha/*nData) * sum[(H-Y)*Xj]
	for it in range (0,MaxIt+1):			
			
		H=[0]*nData  						# h(x), Note: this h is different from first h for each descriptor
		H=pd.DataFrame({'ph':H})			# ph: 'predicted h'  >> for accumulation sum
		for i in range(0,len(X.columns)):
			H['ph']=H['ph']+X[X.columns[i]]*thetas[i]				
		
		J=[0]*nData  						# J(t)
		J=pd.DataFrame({'j':J})				# J: cost function  >> for accumulation sum
		J['j'] = (H['ph']-Y[0])
		
		# modify thetas
		for i in range(0,len(X.columns)):
			L = J['j'] * X[X.columns[i]]
			thetas[i] = thetas[i] - (float(alpha)/nData) * sum(L)
		
		# CostFunction		
			# Problem Definition: J(t)=(1/2*nData) * sum(H-Y)^2	
		Cost_Function = sum((J['j'])**2) / (2.0*nData)			# Note: 2.0 not 2 !!! :)
		MAE = np.mean(abs(J['j']))
		MAE_std = np.std(abs(J['j']))
		# output file
		if it in save_file:
			with open(output_folder+'/MVLR_results_%s.txt'%moment,'a') as file:		
				file.write('it:'+str(it)+'  CostFunction: '+str(Cost_Function)+'\n')
				file.write('it:'+str(it)+'  MAE         : '+str(MAE)+' +/- '+str(MAE_std)+'\n')		
				file.write('\n')
	
	dfTr = pd.DataFrame(thetas, columns=['Coefficients'], index=['x'+str(i) for i in data.columns]+['const'])
	
	with open(output_folder+'/MVLR_results_%s.txt'%moment,'a') as file:
		file.write('==============================================================================\n')
		file.write('Number of samples : '+str(len(X))+'\n')	
		file.write('Number of features : '+str(len(data.columns))+'\n')		
		file.write('\n')
		file.write('Input Features:\n')
		dfTr.to_csv(file,sep='\t')
		file.write('==============================================================================\n')				

	## Evaluation, Test Set, MAE & RMSE
	if len(Data)>1:
		Xt = Data[1]
		Xt['const'] = [1]*len(Xt)
		Xt = Xt.values
		yt = Labels[1][0].values
		
		myt= np.array(thetas) * Xt			# n*m
		predicted_yt = np.array([sum(i) for i in myt])
		
		AE = np.abs(predicted_yt - yt)
		MAE = np.mean(AE)
		MAE_std = np.std(AE)
		MSE = np.mean(AE**2)
		RMSE = np.sqrt(MSE)
		RMSE_std = np.sqrt(np.std(AE**2))			

		dfAE = pd.DataFrame(AE, columns=['Absolute Error'])
				
		with open(output_folder+'/MVLR_results_%s.txt'%moment,'a') as file:
			file.write(' \n')
			file.write('------------------\n')
			file.write(' Test Set Results \n')
			file.write('------------------\n')
			file.write(' \n')
			dfAE.describe().to_csv(file,sep='\t')
			file.write(' \n')
			file.write('Mean Absolute Error +/- Standard Deviation : ' + str(MAE)+' +/- '+str(MAE_std)+'\n')						
			file.write('Root Mean Squared Error +/- Standard Deviation : ' + str(RMSE)+' +/- '+str(RMSE_std)+'\n')						
			file.write('==============================================================================\n')
			file.write('                                     END\n')
			
	return data, labels, logfile, errorfile

###################################################################################################

def lasso(data,labels, logfile, errorfile,argument):
	""" 
		The Lasso is a shrinkage and selection method for linear regression (from Scikitlearn) 
	"""

	## default argument values 
	alpha=1.0; fit_intercept=True; normalize=False; precompute=False; copy_X=True; max_iter=1000; tol=0.0001; warm_start=False; positive=False; random_state=None; selection='cyclic'
	trf = 1.0; scaled = True; coef_ = True
	
	list_args=["alpha", "fit_intercept", "normalize", "precompute", "copy_X", "max_iter", "tol", "warm_start", "positive", "random_state", "selection","trf","scaled","coef_"]

	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in the lasso function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: Lasso\n   alpha = %s\n   fit_intercept = %s\n   normalize = %s\n   precompute = %s\n   copy_X = %s\n   max_iter = %s\n   tol = %s\n   warm_start = %s\n   positive = %s\n   random_state = %s\n   selection = %s\n   trf = %s\n   scaled = %s\n   coef_ = %s\n" %( alpha ,  fit_intercept ,  normalize ,  precompute ,  copy_X ,  max_iter ,  tol ,  warm_start ,  positive ,  random_state ,  selection , trf , scaled, coef_ )	
	print_function_opts(fline,logfile)
	
	## TrS/TeS and normalization
	Data={}
	Labels={}
	if trf < 1 and trf>0:
		Data[0], Labels[0], Data[1], Labels[1] = TrS_TeS(data, labels, trf, scaled)
	elif trf ==1 or trf==0:
		Labels[0] = labels
		if scaled ==True:
			Data[0] = feature_scaled_single(data)
		else:
			Data[0] = copy.deepcopy(data)
			
	## SVR
	from sklearn import linear_model
	## Notice : no selection, no random_state supported by CCR version of scikitlearn
	lasso = linear_model.Lasso( alpha=alpha ,  fit_intercept=fit_intercept ,  normalize=normalize ,  precompute=precompute ,  copy_X=copy_X ,  max_iter=max_iter ,  tol=tol ,  warm_start=warm_start ,  positive=positive )
	
	X,y = Data[0].values, Labels[0][0].values
	r2_training = lasso.fit(X,y).score(X,y)
	
	moment = str(time.ctime())
	with open(output_folder+'/Lasso_results_%s.txt'%moment,'a') as file:
		file.write('                              Lasso Results\n')
		file.write('==============================================================================\n')
		file.write('\n')
		file.write('----------------------\n')
		file.write(' Training Set Results \n')
		file.write('----------------------\n')
		file.write('\n')
		file.write('Number of samples : '+str(len(X))+'\n')	
		file.write('Number of features : '+str(len(data.columns))+'\n')
		file.write('coefficient of determination, R^2: '+str(r2_training)+'\n')
		file.write('==============================================================================\n')
		
	if len(Data)>1:
		Xt = Data[1].values
		yt = Labels[1][0].values
		r2_test = lasso.score(Xt,yt)
		yt_predict = lasso.predict(Xt)
		dif=abs(yt_predict-yt)
		MAE=np.mean(dif)
		MAE_std = np.std(dif)
		MSE=np.mean(dif**2)
		RMSE=np.sqrt(MSE)
		RMSE_std=np.sqrt(np.std(dif**2))
	
		with open(output_folder+'/Lasso_results_%s.txt'%moment,'a') as file:
			file.write(' \n')
			file.write('------------------\n')
			file.write(' Test Set Results \n')
			file.write('------------------\n')
			file.write(' \n')
			file.write('coefficient of determination, R^2: '+str(r2_test)+'\n')			
			file.write('Mean Absolute Error +/- Standard Deviation: '+str(MAE)+' +/- '+str(MAE_std)+'\n')						
			file.write('Root Mean Squared Error +/- Standard Deviation: '+str(RMSE)+' +/- '+str(RMSE_std)+'\n')						
			if coef_ == True:
				file.write('------------------------------------------------------------------------------\n')	
				file.write(' \n')
				file.write('coefficients: \n')
				file.write('%s\n' % lasso.coef_)
				file.write('intercept: %s \n' % lasso.intercept_)
			file.write('==============================================================================\n')
			file.write('                                     END\n')
			
	return data,labels, logfile, errorfile

###################################################################################################

def lassocv(data,labels, logfile, errorfile, argument):
	""" (by Scikitlearn)
		Lasso linear model with iterative fitting along a regularization path. The best model is selected by cross-validation. 
	"""

	## default argument values 
	eps=0.001; n_alphas=100; alphas=None; fit_intercept=True; normalize=False; precompute='auto'; max_iter=1000; tol=0.0001; copy_X=True; cv=None; verbose=False; n_jobs=1; positive=False	
	list_args=["eps", "n_alphas", "alphas", "fit_intercept", "normalize", "precompute", "max_iter", "tol", "copy_X", "cv", "verbose","n_jobs","positive"]

	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in the lassocv function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: LassoCV\n   eps = %s\n   n_alphas = %s\n   alphas = %s\n   fit_intercept = %s\n   normalize = %s\n   precompute = %s\n   max_iter = %s\n   tol = %s\n   copy_X = %s\n   cv = %s\n   verbose = %s\n   n_jobs = %s\n   positive = %s\n" 	%(  eps ,  n_alphas ,  alphas ,  fit_intercept ,  normalize ,  precompute ,  max_iter ,  tol ,  copy_X ,  cv ,  verbose , n_jobs , positive)
	print_function_opts(fline,logfile)
	
	## X,y
	X = data.values
	y = labels[0].values
			
	## SVR
	from sklearn.linear_model import LassoCV
	## Notice : no selection, no random_state supported by CCR version of scikitlearn
	clf = LassoCV(eps=eps, n_alphas=n_alphas, alphas=alphas, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute, max_iter=max_iter, tol=tol, copy_X=copy_X, cv=cv, verbose=verbose, n_jobs=n_jobs, positive=positive)
	
	clf.fit(X,y)
	
	moment = str(time.ctime())
	with open(output_folder+'/LassoCV_results_%s.txt'%moment,'a') as file:
		file.write('                              LassoCV Results\n')
		file.write('==============================================================================\n')
		file.write('\n')
		file.write('--------------------------\n')
		file.write(' Cross Validation Results \n')
		file.write('--------------------------\n')
		file.write('\n')
		file.write('The grid of alphas used for fitting : '+str(clf.alphas_)+'\n')	
		file.write('The amount of penalization chosen by cross validation : '+str(clf.alpha_)+'\n')
		file.write('==============================================================================\n')
		file.write('                                     END\n')
			
	return data,labels, logfile, errorfile

###################################################################################################

def RidgeRegr(data,labels, logfile, errorfile,argument):
	""" 
		Ridge Regression (from Scikitlearn) 
	"""

	## default argument values 
	alpha=1.0; fit_intercept=True; normalize=False; copy_X=True; max_iter=None; tol=0.001; solver='auto'
	trf = 1.0; scaled = True; coef_ = True; cv = 1
	
	list_args=["alpha", "fit_intercept", "normalize", "copy_X", "max_iter", "tol", "solver","trf","scaled","coef_","cv"]

	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in the Ridge function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: Ridge Regression\n   alpha = %s\n   fit_intercept = %s\n   normalize = %s\n   copy_X = %s\n   max_iter = %s\n   tol = %s\n   solver = %s\n   trf = %s\n   scaled = %s\n   coef_ = %s\n   cv = %s\n" %( alpha ,  fit_intercept ,  normalize ,  copy_X ,  max_iter ,  tol ,  solver ,  trf , scaled, coef_,cv )	
	print_function_opts(fline,logfile)
	
	## TrS/TeS and normalization
	Data={}
	Labels={}
	if trf < 1 and trf>0:
		Data[0], Labels[0], Data[1], Labels[1] = TrS_TeS(data, labels, trf, scaled)
	elif trf ==1 or trf==0:
		Labels[0] = labels
		if scaled ==True:
			Data[0] = feature_scaled_single(data)
		else:
			Data[0] = copy.deepcopy(data)
			
	## SVR
	from sklearn.linear_model import Ridge
	## Notice : no selection, no random_state supported by CCR version of scikitlearn
	ridge = Ridge( alpha=alpha ,  fit_intercept=fit_intercept ,  normalize=normalize ,  copy_X=copy_X ,  max_iter=max_iter ,  tol=tol ,  solver=solver )
	
	X,y = Data[0].values, Labels[0][0].values
	r2_training = ridge.fit(X,y).score(X,y)
	
	moment = str(time.ctime())
	with open(output_folder+'/Ridge_results_%s.txt'%moment,'a') as file:
		file.write('                           Ridge Regression Results\n')
		file.write('==============================================================================\n')
		file.write('\n')
		file.write('----------------------\n')
		file.write(' Training Set Results \n')
		file.write('----------------------\n')
		file.write('\n')
		file.write('Number of samples : '+str(len(X))+'\n')	
		file.write('Number of features : '+str(len(data.columns))+'\n')
		file.write('coefficient of determination, R^2: '+str(r2_training)+'\n')
		file.write('==============================================================================\n')
		
	if len(Data)>1:
		Xt = Data[1].values
		yt = Labels[1][0].values
		r2_test = ridge.score(Xt,yt)
		yt_predict = ridge.predict(Xt)
		dif=abs(yt_predict-yt)
		MAE=np.mean(dif)
		MAE_std = np.std(dif)
		MSE=np.mean(dif**2)
		RMSE=np.sqrt(MSE)
		RMSE_std=np.sqrt(np.std(dif**2))
	
		with open(output_folder+'/Ridge_results_%s.txt'%moment,'a') as file:
			file.write(' \n')
			file.write('------------------\n')
			file.write(' Test Set Results \n')
			file.write('------------------\n')
			file.write(' \n')
			file.write('Number of samples : '+str(len(Xt))+'\n')	
			file.write('Number of features : '+str(len(Xt[0]))+'\n')
			file.write('coefficient of determination, R^2: '+str(r2_test)+'\n')			
			file.write('Mean Absolute Error +/- Standard Deviation: '+str(MAE)+' +/- '+str(MAE_std)+'\n')						
			file.write('Root Mean Squared Error +/- Standard Deviation: '+str(RMSE)+' +/- '+str(RMSE_std)+'\n')						
			if coef_ == True:
				file.write('------------------------------------------------------------------------------\n')	
				file.write(' \n')
				file.write('coefficients: \n')
				file.write('%s\n' % ridge.coef_)
			file.write('==============================================================================\n')
			file.write('                                     END\n')
	## Cross Validation
	if cv>1:
		from sklearn.cross_validation import KFold
		kf = KFold(len(data), n_folds=cv, shuffle=True)

		moment = str(time.ctime())
		kcv=0
		mae=[]
		rmse=[]		
		for sample_TrS, sample_TeS in kf:
			kcv+=1	
			TrS_data = data.iloc[sample_TrS,:]
			TrS_label = labels.iloc[sample_TrS,:] 
			TrS_data.index = pd.Index(range(len(TrS_data)))
			TrS_label.index = pd.Index(range(len(TrS_label)))
	
			TeS_data = data.iloc[sample_TeS,:]
			TeS_label = labels.iloc[sample_TeS,:]
			TeS_data.index = pd.Index(range(len(TeS_data)))
			TeS_label.index = pd.Index(range(len(TeS_label)))
	
			if scaled == True:
				TrS_data, TeS_data = feature_scaled(TrS_data, TeS_data)
			
			ridge = Ridge( alpha=alpha ,  fit_intercept=fit_intercept ,  normalize=normalize ,  copy_X=copy_X ,  max_iter=max_iter ,  tol=tol ,  solver=solver )
			X,y = TrS_data.values, TrS_label[0].values
			r2_train = ridge.fit(X,y).score(X,y)
	
			
			with open(output_folder+'/RidgeCV_results_%s.txt'%moment,'a') as file:
				file.write('                           Ridge Regression CV#%s Results\n'% kcv)
				file.write('==============================================================================\n')
				file.write('\n')
				file.write('----------------------\n')
				file.write(' Training Set Results \n')
				file.write('----------------------\n')
				file.write('\n')
				file.write('Number of samples : '+str(len(X))+'\n')	
				file.write('Number of features : '+str(len(X[0]))+'\n')
				file.write('coefficient of determination, R^2: '+str(r2_train)+'\n')
				file.write('==============================================================================\n')
		
			
			Xt = TeS_data.values
			yt = TeS_label[0].values
			r2_test = ridge.score(Xt,yt)
			yt_predict = ridge.predict(Xt)
			dif=abs(yt_predict-yt)
			MAE=np.mean(dif)
			MAE_std = np.std(dif)
			MSE=np.mean(dif**2)
			RMSE=np.sqrt(MSE)
			RMSE_std=np.sqrt(np.std(dif**2))
			
			mae.append(MAE)
			rmse.append(RMSE)
			
			with open(output_folder+'/RidgeCV_results_%s.txt'%moment,'a') as file:
				file.write(' \n')
				file.write('------------------\n')
				file.write(' Test Set Results \n')
				file.write('------------------\n')
				file.write(' \n')
				file.write('Number of samples : %s\n' %str(len(Xt)))	
				file.write('Number of features : %s\n' %str(len(Xt[0])))
				file.write('coefficient of determination, R^2: '+str(r2_test)+'\n')			
				file.write('Mean Absolute Error +/- Standard Deviation: '+str(MAE)+' +/- '+str(MAE_std)+'\n')						
				file.write('Root Mean Squared Error +/- Standard Deviation: '+str(RMSE)+' +/- '+str(RMSE_std)+'\n')						
				file.write('==============================================================================\n')
				file.write(' \n')
				file.write(' \n')
			
		with open(output_folder+'/RidgeCV_results_%s.txt'%moment,'a') as file:
			file.write('\n')
			file.write('----------------------\n')
			file.write('MAEs: %s\n'%str(mae))  
			file.write('Total Mean Absolute Error: %s\n'%str(np.mean(mae)))
			file.write('\n')
			file.write('RMSEs: %s\n'%str(rmse))  
			file.write('Total Root Mean Squared Error: %s\n'%str(np.mean(rmse)))
			file.write('----------------------\n')
			file.write('\n')
			file.write('==============================================================================\n')
			file.write('                                     END\n')
			
	return data,labels, logfile, errorfile

###################################################################################################

def RidgeRegrcv(data,labels, logfile, errorfile, argument):
	""" (by Scikitlearn)
		Ridge Regression, linear model with iterative fitting along a regularization path. The best model is selected by cross-validation. 
	"""

	## default argument values 
	alphas=np.array([ 0.1, 1., 10. ]); fit_intercept=True; normalize=False; scoring=None; cv=None; gcv_mode=None; store_cv_values=False	
	list_args=["alphas", "fit_intercept", "normalize", "scoring", "cv", "gcv_mode", "store_cv_values"]

	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in the ridgeRegrcv function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: ridgeRegrCV\n   alphas = %s\n   fit_intercept = %s\n   normalize = %s\n   scoring = %s\n   cv = %s\n   gcv_mode = %s\n   store_cv_values = %s\n" 	%(  alphas ,  fit_intercept ,  normalize ,  scoring ,  cv ,  gcv_mode ,  store_cv_values  )
	print_function_opts(fline,logfile)
	
	## X,y
	X = data.values
	y = labels[0].values
			
	## SVR
	from sklearn.linear_model import RidgeCV
	## Notice : no selection, no random_state supported by CCR version of scikitlearn
	clf = RidgeCV(alphas=alphas, fit_intercept=fit_intercept, normalize=normalize, scoring=scoring, cv=cv, gcv_mode=gcv_mode, store_cv_values=store_cv_values)
	
	clf.fit(X,y)
	
	moment = str(time.ctime())
	with open(output_folder+'/RidgeRegrCV_results_%s.txt'%moment,'a') as file:
		file.write('                            RidgeRegrCV Results\n')
		file.write('==============================================================================\n')
		file.write('\n')
		file.write('--------------------------\n')
		file.write(' Cross Validation Results \n')
		file.write('--------------------------\n')
		file.write('\n')
		file.write('intercept : '+str(clf.intercept_)+'\n')	
		file.write('The amount of penalization chosen by cross validation : '+str(clf.alpha_)+'\n')
		file.write('==============================================================================\n')
		file.write('                                     END\n')
			
	return data,labels, logfile, errorfile
	
###################################################################################################

def elasticNet(data,labels, logfile, errorfile,argument):
	""" 
		Elastic NET, Linear regression with combined L1 and L2 priors as regularizer. (from Scikitlearn) 
	"""

	## default argument values 
	alpha=1.0; l1_ratio=0.5; fit_intercept=True; normalize=False; precompute=False; copy_X=True; max_iter=1000; tol=0.0001; warm_start=False; positive=False; random_state=None; selection='cyclic'
	trf = 1.0; scaled = True; coef_ = True
	
	list_args=["alpha", "l1_ratio", "fit_intercept", "normalize", "precompute", "copy_X", "max_iter", "tol", "warm_start", "positive", "random_state", "selection","trf","scaled","coef_"]

	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in the Elastic Net function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: Elastic Net\n   alpha = %s\n   l1_ratio = %s\n   fit_intercept = %s\n   normalize = %s\n   precompute = %s\n   copy_X = %s\n   max_iter = %s\n   tol = %s\n   warm_start = %s\n   positive = %s\n   random_state = %s\n   selection = %s\n   trf = %s\n   scaled = %s\n   coef_ = %s\n" %( alpha , l1_ratio, fit_intercept ,  normalize ,  precompute ,  copy_X ,  max_iter ,  tol ,  warm_start ,  positive ,  random_state ,  selection , trf , scaled, coef_ )	
	print_function_opts(fline,logfile)
	
	## TrS/TeS and normalization
	Data={}
	Labels={}
	if trf < 1 and trf>0:
		Data[0], Labels[0], Data[1], Labels[1] = TrS_TeS(data, labels, trf, scaled)
	elif trf ==1 or trf==0:
		Labels[0] = labels
		if scaled ==True:
			Data[0] = feature_scaled_single(data)
		else:
			Data[0] = copy.deepcopy(data)
			
	## SVR
	from sklearn.linear_model import ElasticNet
	## Notice : no selection, no random_state supported by CCR version of scikitlearn
	clf = ElasticNet( alpha=alpha ,  l1_ratio=l1_ratio, fit_intercept=fit_intercept ,  normalize=normalize ,  precompute=precompute ,  copy_X=copy_X ,  max_iter=max_iter ,  tol=tol ,  warm_start=warm_start ,  positive=positive )
	
	X,y = Data[0].values, Labels[0][0].values
	r2_training = clf.fit(X,y).score(X,y)
	
	moment = str(time.ctime())
	with open(output_folder+'/ElasticNet_results_%s.txt'%moment,'a') as file:
		file.write('                            Elastic Net Results\n')
		file.write('==============================================================================\n')
		file.write('\n')
		file.write('----------------------\n')
		file.write(' Training Set Results \n')
		file.write('----------------------\n')
		file.write('\n')
		file.write('Number of samples : '+str(len(X))+'\n')	
		file.write('Number of features : '+str(len(data.columns))+'\n')
		file.write('coefficient of determination, R^2: '+str(r2_training)+'\n')
		file.write('==============================================================================\n')
		
	if len(Data)>1:
		Xt = Data[1].values
		yt = Labels[1][0].values
		r2_test = clf.score(Xt,yt)
		yt_predict = clf.predict(Xt)
		dif=abs(yt_predict-yt)
		MAE=np.mean(dif)
		MAE_std = np.std(dif)
		MSE=np.mean(dif**2)
		RMSE=np.sqrt(MSE)
		RMSE_std=np.sqrt(np.std(dif**2))
	
		with open(output_folder+'/ElasticNet_results_%s.txt'%moment,'a') as file:
			file.write(' \n')
			file.write('------------------\n')
			file.write(' Test Set Results \n')
			file.write('------------------\n')
			file.write(' \n')
			file.write('coefficient of determination, R^2: '+str(r2_test)+'\n')			
			file.write('Mean Absolute Error +/- Standard Deviation: '+str(MAE)+' +/- '+str(MAE_std)+'\n')						
			file.write('Root Mean Squared Error +/- Standard Deviation: '+str(RMSE)+' +/- '+str(RMSE_std)+'\n')						
			if coef_ == True:
				file.write('------------------------------------------------------------------------------\n')	
				file.write(' \n')
				file.write('coefficients: \n')
				file.write('coef_ : %s\n' % clf.coef_)
				file.write(' \n')
				file.write('sparse_coef_ : %s\n' % clf.coef_)				
				file.write('intercept: %s \n' % clf.intercept_)
			file.write('==============================================================================\n')
			file.write('                                     END\n')
			
	return data,labels, logfile, errorfile

###################################################################################################

def ElasticNetcv(data,labels, logfile, errorfile, argument):
	""" (from Scikitlearn)
		Elastic Net model with iterative fitting along a regularization path by cross-validation. 
	"""

	## default argument values 
	l1_ratio=0.5; eps=0.001; n_alphas=100; alphas=None; fit_intercept=True; normalize=False; precompute='auto'; max_iter=1000; tol=0.0001; copy_X=True; cv=None; verbose=0; n_jobs=1; positive=False	
	list_args=["l1_ratio","eps", "n_alphas", "alphas", "fit_intercept", "normalize", "precompute", "max_iter", "tol", "copy_X", "cv", "verbose","n_jobs","positive"]

	## read argument values
	for arg in argument:
		arg = arg.strip().split('=')
		if arg[0].strip() in list_args:
			exec(arg[0].strip() + '=' +arg[1].strip())
		else:
			tmp_str = "Error: unknown option in the ElasticNetCV function."
			sys.exit(tmp_str)
	
	## give out options of this function
	fline = "funnction: ElasticNetCV\n   l1_ratio = %s\n   eps = %s\n   n_alphas = %s\n   alphas = %s\n   fit_intercept = %s\n   normalize = %s\n   precompute = %s\n   max_iter = %s\n   tol = %s\n   copy_X = %s\n   cv = %s\n   verbose = %s\n   n_jobs = %s\n   positive = %s\n" 	%(  l1_ratio, eps ,  n_alphas ,  alphas ,  fit_intercept ,  normalize ,  precompute ,  max_iter ,  tol ,  copy_X ,  cv ,  verbose , n_jobs , positive)
	print_function_opts(fline,logfile)
	
	## X,y
	X = data.values
	y = labels[0].values
			
	## SVR
	from sklearn.linear_model import ElasticNetCV
	## Notice : no selection, no random_state supported by CCR version of scikitlearn
	clf = ElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, alphas=alphas, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute, max_iter=max_iter, tol=tol, copy_X=copy_X, cv=cv, verbose=verbose, n_jobs=n_jobs, positive=positive)
	
	clf.fit(X,y)
	
	moment = str(time.ctime())
	with open(output_folder+'/ElasticNetCV_results_%s.txt'%moment,'a') as file:
		file.write('                            ElasticNetCV Results\n')
		file.write('==============================================================================\n')
		file.write('\n')
		file.write('--------------------------\n')
		file.write(' Cross Validation Results \n')
		file.write('--------------------------\n')
		file.write('\n')
		file.write('The grid of alphas used for fitting : '+str(clf.alphas_)+'\n')	
		file.write('\n')
		file.write('Parameter vector (coef_) : '+str(clf.coef_)+'\n')	
		file.write('intercept : '+str(clf.intercept_)+'\n')	
		file.write('\n')
		file.write('The amount of penalization chosen by cross validation : '+str(clf.alpha_)+'\n')
		file.write('The compromise between l1 and l2 penalization chosen by cross validation : '+str(clf.l1_ratio_)+'\n')
		file.write('==============================================================================\n')
		file.write('                                     END\n')
			
	return data,labels, logfile, errorfile

###################################################################################################

def save_data(data,labels, logfile, errorfile,argument):
	""" 
		save data and labels 
	"""
	
	## give out options of this function
	fline = "funnction: save_data\n"	
	print_function_opts(fline,logfile)
		
	data.to_csv(output_folder+'/newdata_'+str(len(data))+'*'+str(len(data.columns))+'_%s.csv'%time.ctime(),header=None,index=False)
	labels.to_csv(output_folder+'/newlabels_%s.csv'%time.ctime(),header=None, index=False)
	
	return data,labels, logfile, errorfile

###################################################################################################	

def help():
	"""(help):
		list of functions ( options )
	"""
	
	# give out options of this function
	fline = "funnction: help\n"	
	print_function_opts(fline,logfile)
	
	# main
	tmp_str= """
	HELP
	
	Notes about scripting:
	
	*** All the functions and their corresponding options should be specified in a script file.
	
	*** The script file and the main program(CheML.py) must be in a same directory.
	
	*** The format of script file (and only format) must be ".cheml". 
		This file will be recognized by CheML main program automatically. 
	
	*** ONLY lines starting with 'hashtag' would be read by CheML main program.
	
	*** A script file must always have one "initialization" function, which determines the path of input files.
	
	*** The order of functions is important. They will be performed respectively.
	
	===========================================================================================================	
	
	list of functions:	
	
	*** initialization
	mandatory options: 
				data_path : path of input data file, string
				label_path : path of labels file, string
	default options:  
				data_file = None
				log_file = "log.txt"  
				error_file = "error.txt"  
				print_level = 2
				
	END\						
	"""
	print
	print tmp_str
	logfile.write('\n' + tmp_str + '\n\n')
	
	tmp_str = "============================================================================== "
	print
	print tmp_str
	logfile.write('\n' + tmp_str + '\n\n')



#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
"""*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*

 								   	   MAIN SCRIPT	
 										  CheML  		
																						
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#"""
#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*					
	
	
## read script (*.cheml)
time_start = time.time()								
for filename in glob.glob('*.cheml'):
	script = open(filename,'r')
script = script.readlines()
	
## make to-do list
todo={}
todo_order=[]
for line in script:
	if '#' not in line:
		continue
	
	line = line.strip()
	args=[]
	
	line = line.split('%')
	if len(line)>1 :
		for i in range(1,len(line)):
			args.append(line[i].strip())	
	line = line[0].split('#')
	function = line[1].strip()	
	todo[function] = args
	todo_order.append(function)
	
if "help" in todo_order:
	help()	
	
if "initialization" not in todo_order:
	sys.exit("Error:  CheML requires initialization options")						
		
## working directories ('CheML.out' and 'temporary' folders)
check = 'CheML.out'
i = 0
while os.path.exists(check):
	i+=1
	check = 'CheML.out' + str(i)

output_folder = check
os.makedirs(output_folder)

tmp_folder = output_folder +'/temporary'
os.makedirs(tmp_folder)


## make a copy of script in the output folder to be kept as readme.txt file
for filename in glob.glob('*.cheml'):
	shutil.copyfile(filename, output_folder+'/ReadMe.txt')


## initial files
data, labels, logfile, errorfile = initialization(todo['initialization'])
todo_order.remove('initialization')


## options/functions
options = {'pca'  : pca,
           'kpca' : kpca,
           'lda'  : lda,
           'svr'  : svr,
		   'pre_identical' : pre_identical,
		   'pre_linearDep' : pre_linearDep,
		   'save_data'  : save_data,
		   'ols' : ols,
		   'mvlr' : mvlr,
		   'lasso':lasso,
		   'LassoCV' : lassocv,
		   'RidgeRegr' : RidgeRegr,
		   'RidgeRegrCV' : RidgeRegrcv,
		   'ElasticNet' : elasticNet,
		   'ElasticNetCV' : ElasticNetcv
}


## implementing orders
for funct in todo_order:
	data, labels, logfile, errorfile = options[funct](data, labels, logfile, errorfile,todo[funct])


## END: remove temporary files ***********************************************************
tmp_str = tot_exec_time_str(time_start) + "\n" + std_datetime_str()
print tmp_str + 3*'\n'
logfile.write(tmp_str + 4*'\n')
logfile.close()    
errorfile.close()
shutil.rmtree(tmp_folder)






"""
###################################################################################################

def main(opts,commline_list):
    """(main):
        Driver of CheML.
    """
    time_start = time.time()

# TODO: add banner
# TODO: add parser function
    
    return 0    #successful termination of program
    
##################################################################################################

if __name__=="__main__":
    usage_str = "usage: %prog [options] arg"
    version_str = "%prog " + PROGRAM_VERSION
# TODO: replace with argparser
    parser = OptionParser(usage=usage_str, version=version_str)    

    # it is better to sort options by relevance instead of a rigid structure
    parser.add_option('--job', 
                      dest='input_file', 
                      type='string', 
                      default='input.dat', 
                      help='input/job file [default: %default]')


    opts, args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 2:
        sys.exit("You tried to run CheML without options.")
    main(opts,sys.argv)   #numbering of sys.argv is only meaningful if it is launched as main
    
else:
    sys.exit("Sorry, must run as driver...")
    

if __name__ == '__main__':
    pass

"""