# Note: the only pyval with capital initial is: None {Not here, but in the SCRIPT_NAME.xml}
# Note: list of values must be written in the quotation mark
from lxml import objectify, etree

CheML = objectify.Element("CheML",generation_date="12/15/2015", version="0.0.01")

INPUT = objectify.SubElement(CheML, "INPUT", status = 'on')
CheML.INPUT.data_path = "benchmarks/homo_dump/sample_50/data_NOsmi_50.csv"
CheML.INPUT.data_delimiter = 'None'
CheML.INPUT.data_header = 0
CheML.INPUT.data_skiprows = 0
CheML.INPUT.target_path = "benchmarks/homo_dump/sample_50/homo_50.csv"
CheML.INPUT.target_delimiter = 'None'
CheML.INPUT.target_header = 'None'
CheML.INPUT.target_skiprows = 0


OUTPUT = objectify.SubElement(CheML, "OUTPUT", status = 'on')
CheML.OUTPUT.path = "CheML.out"
CheML.OUTPUT.filename_pyscript = "CheML_PyScript.py"
CheML.OUTPUT.filename_logfile = "log.txt"
CheML.OUTPUT.filename_errorfile = "error.txt"

PREPROCESSING = objectify.SubElement(CheML, "PREPROCESSING", status = 'sub')
MISSING_VALUES = objectify.SubElement(PREPROCESSING, "MISSING_VALUES", status = 'on')
CheML.PREPROCESSING.MISSING_VALUES.string_as_null = True
CheML.PREPROCESSING.MISSING_VALUES.missing_values = False
CheML.PREPROCESSING.MISSING_VALUES.inf_as_null = True
CheML.PREPROCESSING.MISSING_VALUES.strategy = "mean"

##########################################################################
objectify.deannotate(CheML)
etree.cleanup_namespaces(CheML)
with open('CMLS.xml', 'w') as outfile:
    outfile.write("%s" %etree.tostring(CheML, pretty_print=True))
