"""
The 'chemml.utils' module includes list_del_indices, std_datetime_str, slurm_script_exclusive,
isfloat, string2nan, 
last modified date: April 25, 2016
"""

from .utilities import list_del_indices
from .utilities import std_datetime_str
from .utilities import tot_exec_time_str
# from .utilities import slurm_script_exclusive
from .utilities import chunk
# from .utilities import choice
# from .utilities import return2Dshape
from .utilities import bool_formatter
from .utilities import padaxis
from .utilities import mol_shapes_to_dims
from .utilities import regression_metrics
from .validation import isfloat
from .validation import islist
from .validation import istuple
from .validation import isnpdot
from .validation import isint
from .validation import value
# from .validation import check_input
from .validation import check_object_col
from .validation import update_default_kwargs

from .utilities import ConvertFile


__all__ = [
    'list_del_indices',
    'std_datetime_str',
    'isfloat',
    'mol_shapes_to_dims',
    'ConvertFile'
]
