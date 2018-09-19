"""
The cheml.visualization module includes (please click on links adjacent to function names for more information):
    - scatter2D: :func:`~cheml.visualization.scatter2D`
    - hist: :func:`~cheml.visualization.hist`
    - decorator: :func:`~cheml.visualization.decorator`
    - SavePlot: :func:`~cheml.visualization.SavePlot`

"""

from .visualization import scatter2D
from .visualization import hist
from .visualization import SavePlot
from .visualization import decorator


__all__ = [
    'scatter2D',
    'hist',
    'decorator',
    'SavePlot',
]
