"""
The 'cheml.preprocessing' module includes missing_values, Imputer_dataframe,
last modified date: Dec. 19, 2015
"""

from .handle_missing import missing_values
from .handle_missing import Imputer_dataframe

from .transform_frame import Scaler_dataframe

__all__ = [
    'missing_values',
    'Imputer_dataframe',
    'Scaler_dataframe',
]