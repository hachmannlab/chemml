from cheml import run

## High refrective index polymers (HRIP) project: density & polarizability & RI
# run(INPUT_FILE = 'input_files/hrip/hrip_descriptor.txt',
#     OUTPUT_DIRECTORY = 'benchmarks/RI_project/liq_org/descriptors/out')
# run(INPUT_FILE = 'input_files/hrip/hrip_NN_gridsearch.txt',
#     OUTPUT_DIRECTORY = 'benchmarks/RI_project/liq_org/results/mlp_k5_MACCS_pol_gridNet')
run(INPUT_FILE = 'input_files/hrip/hrip_NN_learningCurve.txt',
    OUTPUT_DIRECTORY = 'benchmarks/RI_project/liq_org/results/learning_curve/mlp_k10_dragon_d_gridNet')

## Biodegradable polymers (BDP) project: H & G & Transition state
# run(INPUT_FILE = 'input_files/bdp_descriptor.txt',
#     OUTPUT_DIRECTORY = 'benchmarks/BDP_project/results/descriptor')

## Deep eutectic solvents (DES) project: Melting Point


## sykhere's work (app)
# run(INPUT_FILE = 'input_files/bdp_descriptor.txt',
#     OUTPUT_DIRECTORY = 'benchmarks/BDP_project/descriptors/1k')
# run(INPUT_FILE = 'input_files/bdp/bdp_ML_gridsearch.txt',
#    OUTPUT_DIRECTORY = 'benchmarks/BDP_project/results/SVR_cv_HTT_dG')


## DES project
# run(INPUT_FILE = 'input_files/des/des_descriptor.txt',
#     OUTPUT_DIRECTORY = 'benchmarks/DES/MP/descriptors')
# run(INPUT_FILE = 'input_files/des/des_NN_gridsearch.txt',
#     OUTPUT_DIRECTORY = 'benchmarks/DES/MP/results/cv10_NN_dragon')
# run(INPUT_FILE = 'input_files/des/des_classifier_gridsearch.txt',
#     OUTPUT_DIRECTORY = 'benchmarks/DES/MP/results/test_class_dragon')