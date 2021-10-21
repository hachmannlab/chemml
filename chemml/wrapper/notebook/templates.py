def template1():
    """load_cep_homo"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_cep_homo
                    >> homo 0
                    >> smiles 3

                ## (Visualize,plot)
                    << host = chemml
                    << function = hist
                    << color = green
                    << x = 0
                    << bins = 20
                    >> 0 dfx
                    >> fig 1

                ## (Store,figure)
                    << host = chemml
                    << function = SavePlot
                    << kwargs = {'normed':True}
                    << output_directory = plots
                    << filename = homo_histogram
                    >> 2 fig

                ## (Visualize,artist)
                    << host = chemml
                    << function = decorator
                    << title = CEP_HOMO
                    << grid_color = b
                    << xlabel = eHOMO (eV)
                    << ylabel = %
                    << grid = True
                    >> 1 fig
                    >> fig 2

                ## (Enter,python script)
                    << host = chemml
                    << function = PyScript
                    << line01 = print (iv1.head())
                    >> 3 iv1
             """
    return script.strip().split('\n')

def template2():
    """load_cep_homo"""
    script = """
                ## (Store,figure)
                    << host = chemml
                    << function = SavePlot
                    << kwargs = {'normed':True}
                    << output_directory = plots
                    << filename = amwVSdensity
                    >> 0 fig

                ## (Visualize,artist)
                    << host = chemml
                    << function = decorator
                    << title = AMW vs. Density
                    << grid_color = g
                    << xlabel = density (Kg/m3)
                    << ylabel = atomic molecular weight
                    << grid = True
                    << size = 18
                    >> fig 0
                    >> 4 fig

                ## (Enter,python script)
                    << host = chemml
                    << function = PyScript
                    << line01 = print (iv1.head())
                    >> 1 iv1

                ## (Enter,datasets)
                    << host = chemml
                    << function = load_organic_density
                    >> smiles 1
                    >> density 2
                    >> features 3

                ## (Visualize,plot)
                    << host = chemml
                    << function = scatter2D
                    << y = 0
                    << marker = o
                    << x = 'AMW'
                    >> 2 dfy
                    >> 3 dfx
                    >> fig 4
            """
    return script.strip().split('\n')

def template3():
    """load_organic_density"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_xyz_polarizability
                    >> polarizability 0
                    >> coordinates 1

                ## (Visualize,plot)
                    << host = pandas
                    << function = plot
                    << kind = area
                    < subplots = True
                    >> 0 df
                    >> fig 3

                ## (Enter,python script)
                    << host = chemml
                    << function = PyScript
                    << line01 = print(iv1[1])
                    >> 1 iv1

                ## (Store,figure)
                    << host = chemml
                    << function = SavePlot
                    << output_directory = plots
                    << filename = pol_barplot
                    >> 2 fig

                ## (Visualize,artist)
                    << host = chemml
                    << function = decorator
                    << title = polarizability area plot
                    << ylim = (60, None)
                    << grid_linestyle = :
                    << grid = True
                    >> fig 2
                    >> 3 fig
            """
    return script.strip().split('\n')

def template4():
    """CoulombMatrix"""
    script = """
                ## (Enter,datasets)
                << host = chemml
                << function = load_xyz_polarizability
                >> coordinates 0

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = CoulombMatrix
                    >> 0 molecules
                    >> df 1

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << filename = CM_features
                    >> 1 df
            """
    return script.strip().split('\n')

def template5():
    """BagofBonds"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_xyz_polarizability
                    >> coordinates 0

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = BagofBonds
                    >> 0 molecules
                    >> df 1

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << output_directory = descriptors
                    << filename = BoB_features
                    >> 1 df
            """
    return script.strip().split('\n')

def template6():
    """RDKitFingerprint"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_cep_homo
                    >> smiles 2

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = RDKitFingerprint
                    << molfile = @molfile
                    >> 0 molfile
                    >> df 1

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << format = smi
                    << header = False
                    << filename = smiles
                    >> filepath 0
                    >> 2 df

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << filename = Fingerprints
                    >> 1 df
            """
    return script.strip().split('\n')

def template7():
    """Dragon"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_cep_homo
                    >> smiles 0

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << format = smi
                    << header = False
                    << filename = smiles
                    >> 0 df
                    >> filepath 1

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << output_directory = Dragon
                    << filename = Dragon_features
                    >> 2 df

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = Dragon
                    << molFile = @molfile
                    >> 1 molfile
                    >> df 2
            """
    return script.strip().split('\n')

def template8():
    """composition"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_comp_energy
                    >> entries 0
                    >> entries 1
                    >> entries 2
                    >> formation_energy 7

                ## (Represent,inorganic descriptors)
                    << host = chemml
                    << function = ElementFractionAttributeGenerator
                    >> 0 entries
                    >> df 5

                ## (Represent,inorganic descriptors)
                    << host = chemml
                    << function = IonicityAttributeGenerator
                    >> 1 entries
                    >> df 3

                ## (Represent,inorganic descriptors)
                    << host = chemml
                    << function = MeredigAttributeGenerator
                    >> 2 entries
                    >> df 4

                ## (Prepare,data manipulation)
                    << host = pandas
                    << function = concat
                    << axis = 1
                    >> 3 df2
                    >> 4 df3
                    >> 5 df1
                    >> df 6

                ## (Enter,python script)
                    << host = chemml
                    << function = PyScript
                    << line01 = print ('features.shape:', iv1.shape)
                    << line02 = print ('energies.shape:', iv2.shape)
                    >> 6 iv1
                    >> 7 iv2

            """
    return script.strip().split('\n')

def template9():
    """crystal"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_crystal_structures
                    >> entries 0
                    >> entries 1
                    >> entries 5

                ## (Represent,inorganic descriptors)
                    << host = chemml
                    << function = CoordinationNumberAttributeGenerator
                    >> 0 entries
                    >> df 2

                ## (Represent,inorganic descriptors)
                    << host = chemml
                    << function = CoulombMatrixAttributeGenerator
                    >> 1 entries
                    >> df 3

                ## (Prepare,data manipulation)
                    << host = pandas
                    << function = concat
                    << axis = 1
                    >> 2 df2
                    >> 3 df3
                    >> df 4
                    >> 6 df1

                ## (Enter,python script)
                    << host = chemml
                    << function = PyScript
                    << line01 = print ('shape of features:', iv1.shape)
                    >> 4 iv1

                ## (Represent,inorganic descriptors)
                    << host = chemml
                    << function = EffectiveCoordinationNumberAttributeGenerator
                    >> 5 entries
                    >> df 6
            """
    return script.strip().split('\n')

def template11():
    """Model selection"""
    script ="""
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_xyz_polarizability
                    >> coordinates 0
                    >> polarizability 2

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = BagofBonds
                    >> 0 molecules
                    >> df 1

                ## (Prepare,split)
                    << host = sklearn
                    << function = train_test_split
                    << test_size = 0.1
                    << random_state = 0
                    >> 1 dfx
                    >> 2 dfy
                    >> dfx_train 3
                    >> dfy_train 4

                ## (Prepare,scaling)
                    << host = sklearn
                    << function = StandardScaler
                    << func_method = fit_transform
                    >> 3 df
                    >> df 7

                ## (Prepare,scaling)
                    << host = sklearn
                    << function = StandardScaler
                    << func_method = fit_transform
                    >> 4 df
                    >> df 5

                ## (Search,evaluate)
                    << host = sklearn
                    << function = scorer_regression
                    << greater_is_better = False
                    >> scorer 8

                ## (Search,grid)
                    << host = sklearn
                    << function = GridSearchCV
                    << scoring = @scorer
                    << estimator = @estimator
                    << param_grid = {'alpha':[1,0.3,0.1,0.03,0.01]}
                    << cv = @cv
                    >> 5 dfy
                    >> 6 cv
                    >> 7 dfx
                    >> 8 scorer
                    >> cv_results_ 9
                    >> cv_results_ 10
                    >> 11 estimator

                ## (Enter,python script)
                    << host = chemml
                    << function = PyScript
                    << line01 = print (iv1.head(10))
                    >> 9 iv1

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << format = txt
                    << output_directory = gridsearch
                    << header = True
                    << filename = GridSearchCV_results
                    >> 10 df

                ## (Prepare,split)
                    << host = sklearn
                    << function = KFold
                    >> api 6

                ## (Model,regression)
                    << host = sklearn
                    << function = MLPRegressor
                    >> api 11

            """
    return script.strip().split('\n')

def template12():
    """Model selection"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_xyz_polarizability
                    >> coordinates 0
                    >> polarizability 2

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = BagofBonds
                    >> 0 molecules
                    >> df 1

                ## (Prepare,split)
                    << host = sklearn
                    << function = train_test_split
                    << test_size = 0.2
                    << random_state = 0
                    >> 1 dfx
                    >> 2 dfy
                    >> dfx_train 3
                    >> dfy_train 4
                    >> dfx_test 10
                    >> dfy_test 13
                    >> dfy_test 21

                ## (Prepare,scaling)
                    << host = sklearn
                    << function = StandardScaler
                    << func_method = fit_transform
                    >> 3 df
                    >> df 6
                    >> api 9

                ## (Prepare,scaling)
                    << host = sklearn
                    << function = StandardScaler
                    << func_method = fit_transform
                    >> 4 df
                    >> df 5
                    >> api 11

                ## (Search,evaluate)
                    << host = sklearn
                    << function = scorer_regression
                    << greater_is_better = False
                    >> scorer 7

                ## (Model,regression)
                    << host = sklearn
                    << function = KernelRidge
                    << kernel = rbf
                    >> api 15

                ## (Search,grid)
                    << host = sklearn
                    << function = GridSearchCV
                    << scoring = @scorer
                    << param_grid = {'alpha':[1,.3,.1,.03,.01,0.003]}
                    << cv = @cv
                    >> 5 dfy
                    >> 6 dfx
                    >> 7 scorer
                    >> cv_results_ 8
                    >> 15 estimator
                    >> best_estimator_ 17
                    >> 19 cv

                ## (Model,regression)
                    << host = sklearn
                    << function = KernelRidge
                    << func_method = predict
                    >> 16 dfx
                    >> 17 api
                    >> dfy_predict 18

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << format = txt
                    << output_directory = gridsearch
                    << header = True
                    << filename = GridSearchCV_results
                    >> 8 df

                ## (Prepare,scaling)
                    << host = sklearn
                    << function = StandardScaler
                    << func_method = transform
                    >> 9 api
                    >> 10 df
                    >> df 16

                ## (Search,evaluate)
                    << host = sklearn
                    << function = evaluate_regression
                    << r2_score = True
                    << mean_absolute_error = True
                    << median_absolute_error = True
                    << mean_squared_error = True
                    << root_mean_squared_error = True
                    >> 12 dfy_predict
                    >> 13 dfy
                    >> evaluation_results_ 14

                ## (Prepare,scaling)
                    << host = sklearn
                    << function = StandardScaler
                    << func_method = inverse_transform
                    >> 11 api
                    >> df 12
                    >> 18 df
                    >> df 20

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << output_directory = results
                    << filename = test_evaluation
                    >> 14 df

                ## (Prepare,split)
                    << host = sklearn
                    << function = LeaveOneOut
                    >> api 19

                ## (Visualize,plot)
                    << host = chemml
                    << function = scatter2D
                    << y = 0
                    << x = 0
                    >> 20 dfx
                    >> 21 dfy
                    >> fig 22

                ## (Visualize,artist)
                    << host = chemml
                    << function = decorator
                    << title = calculated vs. predicted
                    << grid_color = g
                    << xlabel = predicted polarizability (Bohr3)
                    << grid_linestyle = :
                    << ylabel = calculated polarizability (Bohr3)
                    >> 22 fig
                    >> fig 23

                ## (Store,figure)
                    << host = chemml
                    << function = SavePlot
                    << output_directory = results
                    << filename = test_predVScalc
                    >> 24 fig

                ## (Enter,python script)
                    << host = chemml
                    << function = PyScript
                    << line01 = ax = iv1.axes[0]
                    << line02 = ax.plot([90,120],[90,120],'r-')
                    << line03 = ov1 = ax.figure
                    >> 23 iv1
                    >> ov1 24
            """
    return script.strip().split('\n')

def template13():
    """Generate Morgan fingerprints from SMILES codes"""
    script = """
                ## (Enter,table)
                    << host = pandas
                    << function = read_excel
                    << io = pi_smiles.xlsx
                    << engine = openpyxl
                    >> df 0

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << format = smi
                    << output_directory = .
                    << header = False
                    << filename = smiles
                    >> 0 df
                    >> filepath 1

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = RDKitFingerprint
                    >> 1 molfile
                    >> df 2

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << filename = fingerprints
                    >> 2 df
            """
    return script.strip().split('\n')

def template14():
    """Simple ML workflow"""
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_cep_homo
                    >> smiles 0
                    >> homo 4

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << format = smi
                    << header = False
                    << filename = smiles
                    >> 0 df
                    >> filepath 1

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = RDKitFingerprint
                    >> 1 molfile
                    >> df 2
                    >> df 3

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << filename = fps_rdkfp
                    >> 2 df

                ## (Prepare,split)
                    << host = sklearn
                    << function = train_test_split
                    >> 3 dfx
                    >> 4 dfy
                    >> dfx_train 5
                    >> dfy_train 6
                    >> dfx_test 8
                    >> dfy_test 11

                ## (Model,regression)
                    << host = sklearn
                    << function = MLPRegressor
                    << func_method = fit
                    >> 5 dfx
                    >> 6 dfy
                    >> api 7

                ## (Model,regression)
                    << host = sklearn
                    << function = MLPRegressor
                    << func_method = predict
                    >> 7 api
                    >> 8 dfx
                    >> dfy_predict 9
                    >> dfy_predict 10

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << filename = dfy_predict
                    >> 9 df

                ## (Visualize,plot)
                    << host = chemml
                    << function = scatter2D
                    << x = 0
                    << y = 0
                    >> 10 dfx
                    >> 11 dfy
                    >> fig 12

                ## (Store,figure)
                    << host = chemml
                    << function = SavePlot
                    << filename = dfy_actual_vs_dfy_predict
                    << output_directory = .
                    >> 13 fig

                ## (Visualize,artist)
                    << host = chemml
                    << function = decorator
                    << title = true vs. predicted HOMO energy
                    << xlabel = predicted HOMO energy (eV)
                    << ylabel = true HOMO energy (eV)
                    << grid = True
                    << grid_color = g
                    << size = 18
                    >> 12 fig
                    >> fig 13
             """

    return script.strip().split('\n')

def template15():
    """Genetic Algorithm"""
    script= """
                ## (Input,datasets)
                    << host = chemml
                    << function = load_cep_homo
                    >> smiles 0
                    >> homo 4

                ## (Output,file)
                    << host = chemml
                    << function = SaveFile
                    << format = smi
                    << header = False
                    << filename = smiles
                    >> 0 df
                    >> filepath 1

                ## (Represent,molecular descriptors)
                    << host = chemml
                    << function = RDKitFingerprint
                    >> 1 molfile
                    >> df 2
                    >> df 3

                ## (Output,file)
                    << host = chemml
                    << function = SaveFile
                    << filename = fps_rdkfp
                    >> 2 df

                ## (Prepare,split)
                    << host = sklearn
                    << function = train_test_split
                    >> 3 dfx
                    >> 4 dfy
                    >> dfx_train 5
                    >> dfy_train 6
                    >> dfx_test 7
                    >> dfy_test 8

                ## (Optimize,genetic algorithm)
                    << host = chemml
                    << function = GA
                    << algorithm = 3
                    << ml_model = MLPRegressor
                    << evaluate = ./chemml/chemml/datasets/GA_files/ga_eval.txt
                    << space = ./chemml/chemml/datasets/GA_files/space.txt
                    << error_metric = ./chemml/chemml/datasets/GA_files/error_metric.txt
                    << test_hyperparameters = ./chemml/chemml/datasets/GA_files/test_hyperparameters.txt
                    << single_obj = ./chemml/chemml/datasets/GA_files/single_obj.txt
                    << fitness = (min,)
                    << pop_size = 5
                    << crossover_size = 2
                    << mutation_size = 2
                    << n_splits = 5
                    << n_generations = 5
                    >> 5 dfx_train
                    >> 6 dfy_train
                    >> 7 dfx_test
                    >> 8 dfy_test
                    >> best_ind_df 9
                    >> best_individual 10

                ## (Output,file)
                    << host = chemml
                    << function = SaveFile
                    << format = csv
                    << filename = best_ind_df
                    >> 9 df

                ## (Output,file)
                    << host = chemml
                    << function = SaveFile
                    << format = csv
                    << filename = best_individual
                    >> 10 df
            """
    return script.strip().split('\n')