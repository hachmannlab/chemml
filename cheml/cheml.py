#!/usr/bin/env python

PROGRAM_NAME = "CheML"
PROGRAM_VERSION = "v0.0.1"
REVISION_DATE = "2015-06-02"
AUTHORS = "Johannes Hachmann (hachmann@buffalo.edu) and Mojtaba Haghighatlari (mojtabah@buffalo.edu)"
CONTRIBUTORS = """ """
DESCRIPTION = "CheML is a machine learning and informatics program suite for the chemical and materials sciences."

# Version history timeline (move to CHANGES periodically):
# v0.0.1 (2015-06-02): complete refactoring of original CheML code in new package format


###################################################################################################
# TASKS OF THIS MODULE:
# -main function
###################################################################################################

###################################################################################################
#TODO:
# -restructure more general functions into modules
###################################################################################################

import sys
import os
import time
# TODO: this should at some point replaced with argparser
from optparse import OptionParser


###################################################################################################

def main(opts,commline_list):
    """(main):
        Driver of CheML.
    """
    time_start = time.time()

# TODO: add banner
# TODO: add parser function
    
    return 0    #successful termination of program
    
##################################################################################################

if __name__=="__main__":
    usage_str = "usage: %prog [options] arg"
    version_str = "%prog " + PROGRAM_VERSION
# TODO: replace with argparser
    parser = OptionParser(usage=usage_str, version=version_str)    

    # it is better to sort options by relevance instead of a rigid structure
    parser.add_option('--job', 
                      dest='input_file', 
                      type='string', 
                      default='input.dat', 
                      help='input/job file [default: %default]')


    opts, args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 2:
        sys.exit("You tried to run CheML without options.")
    main(opts,sys.argv)   #numbering of sys.argv is only meaningful if it is launched as main
    
else:
    sys.exit("Sorry, must run as driver...")
    

if __name__ == '__main__':
    pass