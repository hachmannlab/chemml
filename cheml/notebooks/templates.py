def template1():
    script = """## (Enter,xyz)
    << host = cheml
    << function = XYZreader
    << path_pattern = required_required
    >> molecules 0

## (Prepare,feature representation)
    << host = cheml
    << function = Coulomb_Matrix
    >> 0 molecules
    >> df 1

## (Store,file)
    << host = cheml
    << function = SaveFile
    << filename = required_required
    >> 1 df

"""
    return script.strip().split('\n')

def template2():
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

