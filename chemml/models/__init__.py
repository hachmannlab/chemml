"""
The chemml.models.keras module includes (please click on links adjacent to function names for more information):
"""

from chemml.utils.backend import is_tensorflow_available

# from chemml.models.mlp import MLP
from .mlp import MLP
from .graphconvnetwork import NeuralGraphFingerprint
from .transfer import TransferLearning

__all__ = ['MLP', 'NeuralGraphFingerprint', 'TransferLearning']

# NeuralGraphHidden/NeuralGraphOutput are TensorFlow-only Keras layers; only
# expose them when TensorFlow is actually importable so chemml.models can
# still be imported with a pytorch-only environment.
if is_tensorflow_available():
    from .graphconvlayers import NeuralGraphHidden, NeuralGraphOutput
    __all__ += ['NeuralGraphHidden', 'NeuralGraphOutput']