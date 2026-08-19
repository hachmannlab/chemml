"""
The chemml.visualization module includes (please click on links adjacent to function names for more information):
    - scatter2D: :func:`~chemml.visualization.scatter2D`
    - hist: :func:`~chemml.visualization.hist`
    - decorator: :func:`~chemml.visualization.decorator`
    - SavePlot: :func:`~chemml.visualization.SavePlot`
    - ClassificationPlots: :func:`~chemml.visualization.ClassificationPlots`

"""

from .visualization import scatter2D
from .visualization import hist
from .visualization import SavePlot
from .visualization import decorator
from .visualization import ClassificationPlots


__all__ = [
    'scatter2D',
    'hist',
    'decorator',
    'SavePlot',
    'ClassificationPlots',
]
