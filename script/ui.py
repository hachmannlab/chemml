# Note: the only pyval with capital initial is: None {Not here, but in the SCRIPT_NAME.xml}
# Note: list of values must be written in the quotation mark
# Note: sth like np.float should be wrapped by quote in here
from lxml import objectify, etree
import numpy as np
from sct_utils import std_datetime_str

cmls = objectify.Element("CheML", date=std_datetime_str('date'), time=std_datetime_str('time'), version="1.1.0")

f1 = objectify.SubElement(cmls, "f1", function = "INPUT", status = 'on')
cmls.f1.data_path = "benchmarks/homo_dump/sample_50/data_NOsmi_50.csv"
cmls.f1.data_delimiter = 'None'
cmls.f1.data_header = 0
cmls.f1.data_skiprows = 0
cmls.f1.target_path = "benchmarks/homo_dump/sample_50/homo_50.csv"
cmls.f1.target_delimiter = 'None'
cmls.f1.target_header = 'None'
cmls.f1.target_skiprows = 0

f2 = objectify.SubElement(cmls, "f2", function = "OUTPUT", status = 'on')
cmls.f2.path = "CheML.out"
cmls.f2.filename_pyscript = "CheML_PyScript.py"
cmls.f2.filename_logfile = "log.txt"
cmls.f2.filename_errorfile = "error.txt"

f3 = objectify.SubElement(cmls, "f3", function = "MISSING_VALUES", status = 'on')
cmls.f3.string_as_null = True
cmls.f3.missing_values = False
cmls.f3.inf_as_null = True
cmls.f3.strategy = "mean"

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
cmls.f8.norm = 'l2'
cmls.f8.copy = True

f9 = objectify.SubElement(cmls, "f9", function = "Binarizer", status = 'on')
cmls.f9.threshold = 0
cmls.f9.copy = True

f10 = objectify.SubElement(cmls, "f10", function = "OneHotEncoder", status = 'on')
cmls.f10.n_values = 'auto'
cmls.f10.categorical_features = 'all'
cmls.f10.dtype = 'np.float'
cmls.f10.sparse = True
cmls.f10.handle_unknown = 'error'

f11 = objectify.SubElement(cmls, "f11", function = "PolynomialFeatures", status = 'on')
cmls.f11.degree = 2
cmls.f11.interaction_only = False
cmls.f11.include_bias = True

f12 = objectify.SubElement(cmls, "f12", function = "FunctionTransformer", status = 'on')
cmls.f12.func = 'None'
cmls.f12.validate = True
cmls.f12.accept_sparse = False
cmls.f12.pass_y = False


##########################################################################
objectify.deannotate(cmls)
etree.cleanup_namespaces(cmls)
with open('CMLS.xml', 'w') as outfile:
    outfile.write("%s" %etree.tostring(cmls, pretty_print=True))