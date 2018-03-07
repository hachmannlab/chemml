#! /usr/bin/env python

import argparse
from cheml import wrapperRUN

parser = argparse.ArgumentParser(description="ChemML will be started by specifying a script file as a todo list")
parser.add_argument("-i", type=str, required=True,
                    help="input directory: must include the input file name and its format")
parser.add_argument("-o", type=str, required=True,
                    help="output directory: must be unique otherwise incrementing numbers will be added")
args = parser.parse_args()
SCRIPT_NAME = args.i
output_directory = args.o

wrapperRUN(INPUT_FILE = SCRIPT_NAME, OUTPUT_DIRECTORY = output_directory)

