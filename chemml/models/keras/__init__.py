"""
The chemml.models.keras module includes (please click on links adjacent to function names for more information):
    - MLP: :func:`~chemml.nn.keras.MLP`
    - MLP_sklearn: :func:`~chemml.nn.keras.MLP_sklearn`
"""


from .mlp import MLP
from .mlp import MLP_sklearn




__all__ = [
    'MLP',
    'MLP_sklearn'
    ]