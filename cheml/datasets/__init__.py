"""
The cheml.datasets module includes:
    - load_cep_homo: :func:`~cheml.datasets.load_cep_homo`
    - load_organic_density: :func:`~cheml.datasets.load_organic_density`
    - load_xyz_polarizability: :func:`~cheml.datasets.load_xyz_polarizability`
"""

from .base import load_cep_homo
from .base import load_organic_density
from .base import load_xyz_polarizability


__all__ = [
    'load_cep_homo',
    'load_organic_density',
    'load_xyz_polarizability'
]


# if chemml is not installed yet:
# import sys
# sys.path.insert(0, "/Users/mojtabahaghighatlari/Box Sync/Hachmann_Group/7_MLpackages/chemml")
# sys.path.insert(0, "/projects/academic/hachmann/mojtaba/chemml/")
