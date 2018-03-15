def template1():
    """CoulombMatrix"""
    script = """
                ## (Enter,datasets)
                << host = cheml
                << function = load_xyz_polarizability
                >> coordinates 0

                ## (Represent,molecular descriptors)
                    << host = cheml
                    << function = CoulombMatrix
                    >> 0 molecules
                    >> df 1

                ## (Store,file)
                    << host = cheml
                    << function = SaveFile
                    << filename = CM_features
                    >> 1 df
            """
    return script.strip().split('\n')

def template2():
    """BagofBonds"""
    script = """
                ## (Enter,datasets)
                    << host = cheml
                    << function = load_xyz_polarizability
                    >> coordinates 0

                ## (Represent,molecular descriptors)
                    << host = cheml
                    << function = BagofBonds
                    >> 0 molecules
                    >> df 1

                ## (Store,file)
                    << host = cheml
                    << function = SaveFile
                    << output_directory = descriptors
                    << filename = BoB_features
                    >> 1 df
            """
    return script.strip().split('\n')

def template3():
    """RDKitFingerprint"""
    script = """
                ## (Enter,datasets)
                    << host = cheml
                    << function = load_cep_homo
                    >> smiles 2

                ## (Represent,molecular descriptors)
                    << host = cheml
                    << function = RDKitFingerprint
                    << molfile = @molfile
                    >> 0 molfile
                    >> df 1

                ## (Store,file)
                    << host = cheml
                    << function = SaveFile
                    << format = smi
                    << header = False
                    << filename = smiles
                    >> filepath 0
                    >> 2 df

                ## (Store,file)
                    << host = cheml
                    << function = SaveFile
                    << filename = Fingerprints
                    >> 1 df
            """
    return script.strip().split('\n')

def template4():
    """Dragon"""
    script = """
                ## (Enter,datasets)
                    << host = cheml
                    << function = load_cep_homo
                    >> smiles 0

                ## (Store,file)
                    << host = cheml
                    << function = SaveFile
                    << format = smi
                    << header = False
                    << filename = smiles
                    >> 0 df
                    >> filepath 1

                ## (Store,file)
                    << host = cheml
                    << function = SaveFile
                    << output_directory = Dragon
                    << filename = Dragon_features
                    >> 2 df

                ## (Represent,molecular descriptors)
                    << host = cheml
                    << function = Dragon
                    << molFile = @molfile
                    >> 1 molfile
                    >> df 2
            """
    return script.strip().split('\n')


def template7():
    """Dragon"""
    script = """
                ## (Enter,datasets)
                    << host = cheml
                    << function = load_cep_homo
                    >> homo 0
                    >> smiles 1

                ## (Visualize,Plot)
                    << host = cheml
                    << function = hist
                    << nbins = 20
                    << rwidth = 0.8
                    << ylabel = numbers
                    << xlabel = HOMO_eV
                    >> 0 df1
                    >> fig 2

                ## (Enter,python script)
                    << host = cheml
                    << function = PyScript
                    << line01 = print iv1.head()
                    >> 1 iv1

                ## (Store,figure)
                    << host = cheml
                    << function = SaveFigure
                    << filename = homo_hist
                    >> 2 fig

            """
    return script.strip().split('\n')

def template20():
    script ="""

        ## (Prepare,split)
            << host = sklearn
            << function = train_test_split
            << test_size = 0.1
            << random_state = 0
            >> dfx_train 0
            >> dfy_train 1

        ## (Prepare,scale)
            << host = sklearn
            << function = StandardScaler
            << func_method = fit_transform
            >> 0 df
            >> df 4

        ## (Prepare,scale)
            << host = sklearn
            << function = StandardScaler
            << func_method = fit_transform
            >> 1 df
            >> df 3

        ## (Search,evaluate)
            << host = sklearn
            << function = scorer_regression
            << greater_is_better = False
            >> scorer 2

        ## (Model,regression)
            << host = sklearn
            << function = MLPRegressor
            << validation_fraction = 0.1
            << early_stopping = True
            >> api 5

        ## (Search,grid)
            << host = sklearn
            << function = GridSearchCV
            << scoring = @scorer
            << estimator = @estimator
            << param_grid = {'alpha':[3,1,.3,.1,.03,.01]}
            << cv = 3
            >> 2 scorer
            >> 3 dfy
            >> 4 dfx
            >> 5 estimator
            >> cv_results_ 6
            >> cv_results_ 7

        ## (Enter,python script)
            << host = cheml
            << function = PyScript
            << line01 = print iv1.head(10)
            >> 6 iv1

        ## (Store,file)
            << host = cheml
            << function = SaveFile
            << format = txt
            << header = True
            << filename = GridSearchCV_results
            >> 7 df


    """
    return script.strip().split('\n')

