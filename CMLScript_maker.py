# Note: the only pyval with capital initial is: None {Not here, but in the SCRIPT_NAME.xml}
# Note: list of values must be written in the quotation mark
from lxml import objectify, etree

ChemML = objectify.Element("ChemML",generation_date="12/15/2015", version="0.0.01")

INPUT = objectify.SubElement(ChemML, "INPUT", status = 'on')
ChemML.INPUT.data_path = "benchmarks/homo_dump/sample_50/data_NOsmi_50.csv"
ChemML.INPUT.data_delimiter = 'None'
ChemML.INPUT.data_header = 0
ChemML.INPUT.data_skiprows = 0
ChemML.INPUT.target_path = "benchmarks/homo_dump/sample_50/homo_50.csv"
ChemML.INPUT.target_delimiter = 'None'
ChemML.INPUT.target_header = 'None'
ChemML.INPUT.target_skiprows = 0


OUTPUT = objectify.SubElement(ChemML, "OUTPUT", status = 'on')
ChemML.OUTPUT.path = "ChemML.out"
ChemML.OUTPUT.filename_pyscript = "ChemML_PyScript.py"
ChemML.OUTPUT.filename_logfile = "log.txt"
ChemML.OUTPUT.filename_errorfile = "error.txt"

PREPROCESSING = objectify.SubElement(ChemML, "PREPROCESSING", status = 'sub')
MISSING_VALUES = objectify.SubElement(PREPROCESSING, "MISSING_VALUES", status = 'on')
ChemML.PREPROCESSING.MISSING_VALUES.string_as_null = True
ChemML.PREPROCESSING.MISSING_VALUES.missing_values = False
ChemML.PREPROCESSING.MISSING_VALUES.inf_as_null = True
ChemML.PREPROCESSING.MISSING_VALUES.strategy = "mean"

##########################################################################
objectify.deannotate(ChemML)
etree.cleanup_namespaces(ChemML)
with open('SCRIPT_NAME.xml', 'w') as outfile:
    outfile.write("%s" %etree.tostring(ChemML, pretty_print=True))
