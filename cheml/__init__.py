# __name__ = "cheml"
__version__ = "0.4.2"
__author__ = ["Mojtaba Haghighatlari (mojtabah@buffalo.edu)", "Johannes Hachmann (hachmann@buffalo.edu)"]

from .wrappers.engine import run as wrapperRUN
# from cheml.notebooks import wrapperGUI

import sys
sys.dont_write_bytecode = True
