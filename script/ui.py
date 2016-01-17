# Note: the only pyval with capital initial is: None {Not here, but in the SCRIPT_NAME.xml}
# Note: list of values must be written in the quotation mark
# Note: sth like np.float should be wrapped by quote in here
from lxml import objectify, etree
from sct_utils import std_datetime_str

cmls = objectify.Element("CheML", date=std_datetime_str('date'), time=std_datetime_str('time'), version="1.1.0")

f1 = objectify.SubElement(cmls, "f1", function = "INPUT", status = 'on')
cmls.f1.data_path = "'benchmarks/homo_dump/sample_50/data_NOsmi_50.csv'"
cmls.f1.data_delimiter = 'None'
cmls.f1.data_header = 0
cmls.f1.data_skiprows = 0
cmls.f1.target_path = "'benchmarks/homo_dump/sample_50/homo_50.csv'"
cmls.f1.target_delimiter = 'None'
cmls.f1.target_header = 'None'
cmls.f1.target_skiprows = 0

f2 = objectify.SubElement(cmls, "f2", function = "OUTPUT", status = 'on')
cmls.f2.path = "'CheML.out'"
cmls.f2.filename_pyscript = "CheML_PyScript.py"
cmls.f2.filename_logfile = "'log.txt'"
cmls.f2.filename_errorfile = "'error.txt'"

f3 = objectify.SubElement(cmls, "f3", function = "MISSING_VALUES", status = 'on')
cmls.f3.string_as_null = True
cmls.f3.missing_values = False
cmls.f3.inf_as_null = True
cmls.f3.strategy = "'mean'"

f4 = objectify.SubElement(cmls, "f4", function = "StandardScaler", status = 'on')
cmls.f4.copy = True
cmls.f4.with_mean = True
cmls.f4.with_std = True

f5 = objectify.SubElement(cmls, "f5", function = "MinMaxScaler", status = 'on')
cmls.f5.feature_range = '(0,1)'
cmls.f5.copy = True

f6 = objectify.SubElement(cmls, "f6", function = "MaxAbsScaler", status = 'on')
cmls.f6.copy = True

f7 = objectify.SubElement(cmls, "f7", function = "RobustScaler", status = 'on')
cmls.f7.with_centering = True
cmls.f7.with_scaling = True
cmls.f7.copy = True

f8 = objectify.SubElement(cmls, "f8", function = "Normalizer", status = 'on')
cmls.f8.norm = "'l2'"
cmls.f8.copy = True

f9 = objectify.SubElement(cmls, "f9", function = "Binarizer", status = 'on')
cmls.f9.threshold = 0.0
cmls.f9.copy = True

f10 = objectify.SubElement(cmls, "f10", function = "OneHotEncoder", status = 'on')
cmls.f10.n_values = "'auto'"
cmls.f10.categorical_features = "'all'"
cmls.f10.dtype = 'np.float'
cmls.f10.sparse = True
cmls.f10.handle_unknown = "'error'"

f11 = objectify.SubElement(cmls, "f11", function = "PolynomialFeatures", status = 'on')
cmls.f11.degree = 2
cmls.f11.interaction_only = False
cmls.f11.include_bias = True

f12 = objectify.SubElement(cmls, "f12", function = "FunctionTransformer", status = 'on')
cmls.f12.func = 'None'
cmls.f12.validate = True
cmls.f12.accept_sparse = False
cmls.f12.pass_y = False

f13 = objectify.SubElement(cmls, "f13", function = "VarianceThreshold", status = 'on')
cmls.f13.threshold = 0.0

f14 = objectify.SubElement(cmls, "f14", function = "SelectKBest", status = 'on')
cmls.f14.score_func = 'f_regression'
cmls.f14.k = 10

f15 = objectify.SubElement(cmls, "f15", function = "SelectPercentile", status = 'on')
cmls.f15.score_func = 'f_regression'
cmls.f15.percentile = 10

f16 = objectify.SubElement(cmls, "f16", function = "SelectFpr", status = 'on')
cmls.f16.score_func = 'f_regression'
cmls.f16.alpha = 0.05

f17 = objectify.SubElement(cmls, "f17", function = "SelectFdr", status = 'on')
cmls.f17.score_func = 'f_regression'
cmls.f17.alpha = 0.05

f18 = objectify.SubElement(cmls, "f18", function = "SelectFwe", status = 'on')
cmls.f18.score_func = 'f_regression'
cmls.f18.alpha = 0.05

f19 = objectify.SubElement(cmls, "f19", function = "RFE", status = 'on')
cmls.f19.estimator = '' # This must be called sooner than this function, no worries about running this once beforehand!
cmls.f19.n_features_to_select = 'None'
cmls.f19.step = 1
cmls.f19.estimator_params = 'None'
cmls.f19.verbose = 0


f20 = objectify.SubElement(cmls, "f20", function = "RFECV", status = 'on')
cmls.f20.estimator = '' # This must be called sooner than this function, no worries about running this once beforehand!
cmls.f20.step = 1
cmls.f20.cv = 'None'
cmls.f20.scoring = 'None'
cmls.f20.estimator_params = 'None'
cmls.f20.verbose = 0

##########################################################################
objectify.deannotate(cmls)
etree.cleanup_namespaces(cmls)
with open('CMLS.xml', 'w') as outfile:
    outfile.write("%s" %etree.tostring(cmls, pretty_print=True))