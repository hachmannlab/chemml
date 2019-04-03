"""
The 'chemml.models' module includes (please click on links adjacent to function names for more information):
    - OrganicLorentzLorenz: :func:`~chemml.models.keras.trained.OrganicLorentzLorenz`
    - MLP: :func:`~chemml.models.keras.mlp.MLP`
"""

from chemml.models.keras.mlp import MLP
from chemml.models.keras.trained import OrganicLorentzLorenz


__all__ = [
    'OrganicLorentzLorenz',
    'MLP'
]
