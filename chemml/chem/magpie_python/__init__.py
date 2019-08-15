"""
This module is adapted entirely from Magpie (https://bitbucket.org/wolverton/magpie).
If you are using this module, please cite Magpie as:
 L. Ward, A. Agrawal, A. Choudhary, and C. Wolverton, "A general-purpose machine learning framework for predicting properties of inorganic materials," npj Computational Materials, vol. 2, no. 1, Aug. 2016.
For more information regarding the python version of Magpie, please see https://github.com/ramv2/magpie_python.
The chemml.chem.magpie_python module includes (please click on links adjacent to function names for more information):
    - APEAttributeGenerator: :func:`~chemml.chem.magpie_python.APEAttributeGenerator`
    - ChargeDependentAttributeGenerator: :func:`~chemml.chem.magpie_python.ChargeDependentAttributeGenerator`
    - ElementalPropertyAttributeGenerator: :func:`~chemml.chem.magpie_python.ElementalPropertyAttributeGenerator`
    - ElementFractionAttributeGenerator: :func:`~chemml.chem.magpie_python.ElementFractionAttributeGenerator`
    - ElementPairPropertyAttributeGenerator: :func:`~chemml.chem.magpie_python.ElementPairPropertyAttributeGenerator`
    
    - GCLPAttributeGenerator: :func:`~chemml.chem.magpie_python.GCLPAttributeGenerator`
    - IonicCompoundProximityAttributeGenerator: :func:`~chemml.chem.magpie_python.IonicCompoundProximityAttributeGenerator`
    - IonicityAttributeGenerator: :func:`~chemml.chem.magpie_python.IonicityAttributeGenerator`
    - MeredigAttributeGenerator: :func:`~chemml.chem.magpie_python.MeredigAttributeGenerator`
    - StoichiometricAttributeGenerator: :func:`~chemml.chem.magpie_python.StoichiometricAttributeGenerator`
    
    - ValenceShellAttributeGenerator: :func:`~chemml.chem.magpie_python.ValenceShellAttributeGenerator`
    - YangOmegaAttributeGenerator: :func:`~chemml.chem.magpie_python.YangOmegaAttributeGenerator`
    - APRDFAttributeGenerator: :func:`~chemml.chem.magpie_python.APRDFAttributeGenerator`
    - ChemicalOrderingAttributeGenerator: :func:`~chemml.chem.magpie_python.ChemicalOrderingAttributeGenerator`
    - CoordinationNumberAttributeGenerator: :func:`~chemml.chem.magpie_python.CoordinationNumberAttributeGenerator`
    
    - CoulombMatrixAttributeGenerator: :func:`~chemml.chem.magpie_python.CoulombMatrixAttributeGenerator`
    - EffectiveCoordinationNumberAttributeGenerator: :func:`~chemml.chem.magpie_python.EffectiveCoordinationNumberAttributeGenerator`
    - LatticeSimilarityAttributeGenerator: :func:`~chemml.chem.magpie_python.LatticeSimilarityAttributeGenerator`
    - LocalPropertyDifferenceAttributeGenerator: :func:`~chemml.chem.magpie_python.LocalPropertyDifferenceAttributeGenerator`
    - LocalPropertyVarianceAttributeGenerator: :func:`~chemml.chem.magpie_python.LocalPropertyVarianceAttributeGenerator`

    - PackingEfficiencyAttributeGenerator: :func:`~chemml.chem.magpie_python.PackingEfficiencyAttributeGenerator`
    - PRDFAttributeGenerator: :func:`~chemml.chem.magpie_python.PRDFAttributeGenerator`
    - StructuralHeterogeneityAttributeGenerator: :func:`~chemml.chem.magpie_python.StructuralHeterogeneityAttributeGenerator`
    - CompositionEntry: :func:`~chemml.chem.magpie_python.CompositionEntry`
    - CrystalStructureEntry: :func:`~chemml.chem.magpie_python.CrystalStructureEntry`
"""
__all__ = [
    'APEAttributeGenerator',
    'ChargeDependentAttributeGenerator',
    'ElementalPropertyAttributeGenerator',
    'ElementFractionAttributeGenerator',
    'ElementPairPropertyAttributeGenerator',
    
    'GCLPAttributeGenerator',
    'IonicCompoundProximityAttributeGenerator',
    'IonicityAttributeGenerator',
    'MeredigAttributeGenerator',
    'StoichiometricAttributeGenerator',
    
    'ValenceShellAttributeGenerator',
    'YangOmegaAttributeGenerator',
    'APRDFAttributeGenerator',
    'ChemicalOrderingAttributeGenerator',
    'CoordinationNumberAttributeGenerator',
    
    'CoulombMatrixAttributeGenerator',
    'EffectiveCoordinationNumberAttributeGenerator',
    'LatticeSimilarityAttributeGenerator',
    'LocalPropertyDifferenceAttributeGenerator',
    'LocalPropertyVarianceAttributeGenerator',

    'PackingEfficiencyAttributeGenerator',
    'PRDFAttributeGenerator',
    'StructuralHeterogeneityAttributeGenerator',
    'CompositionEntry',
    'CrystalStructureEntry',
]
from chemml.chem.magpie_python.attributes.generators.composition.APEAttributeGenerator import \
    APEAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition.ChargeDependentAttributeGenerator \
    import ChargeDependentAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition.ElementalPropertyAttributeGenerator \
    import ElementalPropertyAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition.ElementFractionAttributeGenerator \
    import ElementFractionAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition.ElementPairPropertyAttributeGenerator \
    import ElementPairPropertyAttributeGenerator


from chemml.chem.magpie_python.attributes.generators.composition.GCLPAttributeGenerator import \
    GCLPAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition\
    .IonicCompoundProximityAttributeGenerator import \
    IonicCompoundProximityAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition.IonicityAttributeGenerator import \
    IonicityAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition.MeredigAttributeGenerator import \
    MeredigAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition.StoichiometricAttributeGenerator \
    import StoichiometricAttributeGenerator


from chemml.chem.magpie_python.attributes.generators.composition.ValenceShellAttributeGenerator import \
    ValenceShellAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.composition.YangOmegaAttributeGenerator import \
    YangOmegaAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal.APRDFAttributeGenerator import \
    APRDFAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal.ChemicalOrderingAttributeGenerator import \
    ChemicalOrderingAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal.CoordinationNumberAttributeGenerator \
    import CoordinationNumberAttributeGenerator


from chemml.chem.magpie_python.attributes.generators.crystal.CoulombMatrixAttributeGenerator import \
    CoulombMatrixAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal\
    .EffectiveCoordinationNumberAttributeGenerator import \
    EffectiveCoordinationNumberAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal.LatticeSimilarityAttributeGenerator import\
    LatticeSimilarityAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal.LocalPropertyDifferenceAttributeGenerator \
    import  LocalPropertyDifferenceAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal.LocalPropertyVarianceAttributeGenerator \
    import LocalPropertyVarianceAttributeGenerator


from chemml.chem.magpie_python.attributes.generators.crystal.PackingEfficiencyAttributeGenerator import\
    PackingEfficiencyAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal.PRDFAttributeGenerator import \
    PRDFAttributeGenerator
from chemml.chem.magpie_python.attributes.generators.crystal.StructuralHeterogeneityAttributeGenerator \
    import StructuralHeterogeneityAttributeGenerator

from chemml.chem.magpie_python.data.materials.CompositionEntry import CompositionEntry
from chemml.chem.magpie_python.data.materials.CrystalStructureEntry import CrystalStructureEntry

# __all__ = ["attributes", "data", "models", "test", "utility", "vassal"]
