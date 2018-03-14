"""
The cheml.visualization module includes:
    - Scatter_2D: :func:`~cheml.visualization.Scatter_2D`
    - hist: :func:`~cheml.visualization.hist`
    - SaveFigure: :func:`~cheml.visualization.SaveFigure`


"""

from .visualization import Scatter_2D
from .visualization import hist
from .visualization import SaveFigure


__all__ = [
    'scatter_2D',
    'hist',
    'SaveFigure',
]
