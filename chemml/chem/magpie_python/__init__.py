"""
This module is adapted entirely from Magpie (https://bitbucket.org/wolverton/magpie).
If you are using this module, please cite Magpie as:
 L. Ward, A. Agrawal, A. Choudhary, and C. Wolverton, "A general-purpose machine learning framework for predicting properties of inorganic materials," npj Computational Materials, vol. 2, no. 1, Aug. 2016.
For more information regarding the python version of Magpie, please see https://github.com/ramv2/magpie_python.
The cheml.chem.magpie_python module includes (please click on links adjacent to function names for more information):
    - APEAttributeGenerator: :func:`~cheml.chem.magpie_python.APEAttributeGenerator`
    - ChargeDependentAttributeGenerator: :func:`~cheml.chem.magpie_python.ChargeDependentAttributeGenerator`
    - ElementalPropertyAttributeGenerator: :func:`~cheml.chem.magpie_python.ElementalPropertyAttributeGenerator`
    - ElementFractionAttributeGenerator: :func:`~cheml.chem.magpie_python.ElementFractionAttributeGenerator`
    - ElementPairPropertyAttributeGenerator: :func:`~cheml.chem.magpie_python.ElementPairPropertyAttributeGenerator`
    
    - GCLPAttributeGenerator: :func:`~cheml.chem.magpie_python.GCLPAttributeGenerator`
    - IonicCompoundProximityAttributeGenerator: :func:`~cheml.chem.magpie_python.IonicCompoundProximityAttributeGenerator`
    - IonicityAttributeGenerator: :func:`~cheml.chem.magpie_python.IonicityAttributeGenerator`
    - MeredigAttributeGenerator: :func:`~cheml.chem.magpie_python.MeredigAttributeGenerator`
    - StoichiometricAttributeGenerator: :func:`~cheml.chem.magpie_python.StoichiometricAttributeGenerator`
    
    - ValenceShellAttributeGenerator: :func:`~cheml.chem.magpie_python.ValenceShellAttributeGenerator`
    - YangOmegaAttributeGenerator: :func:`~cheml.chem.magpie_python.YangOmegaAttributeGenerator`
    - APRDFAttributeGenerator: :func:`~cheml.chem.magpie_python.APRDFAttributeGenerator`
    - ChemicalOrderingAttributeGenerator: :func:`~cheml.chem.magpie_python.ChemicalOrderingAttributeGenerator`
    - CoordinationNumberAttributeGenerator: :func:`~cheml.chem.magpie_python.CoordinationNumberAttributeGenerator`
    
    - CoulombMatrixAttributeGenerator: :func:`~cheml.chem.magpie_python.CoulombMatrixAttributeGenerator`
    - EffectiveCoordinationNumberAttributeGenerator: :func:`~cheml.chem.magpie_python.EffectiveCoordinationNumberAttributeGenerator`
    - LatticeSimilarityAttributeGenerator: :func:`~cheml.chem.magpie_python.LatticeSimilarityAttributeGenerator`
    - LocalPropertyDifferenceAttributeGenerator: :func:`~cheml.chem.magpie_python.LocalPropertyDifferenceAttributeGenerator`
    - LocalPropertyVarianceAttributeGenerator: :func:`~cheml.chem.magpie_python.LocalPropertyVarianceAttributeGenerator`

    - PackingEfficiencyAttributeGenerator: :func:`~cheml.chem.magpie_python.PackingEfficiencyAttributeGenerator`
    - PRDFAttributeGenerator: :func:`~cheml.chem.magpie_python.PRDFAttributeGenerator`
    - StructuralHeterogeneityAttributeGenerator: :func:`~cheml.chem.magpie_python.StructuralHeterogeneityAttributeGenerator`
    - CompositionEntry: :func:`~cheml.chem.magpie_python.CompositionEntry`
    - CrystalStructureEntry: :func:`~cheml.chem.magpie_python.CrystalStructureEntry`
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
from attributes.generators.composition.APEAttributeGenerator import \
    APEAttributeGenerator
from attributes.generators.composition.ChargeDependentAttributeGenerator \
    import ChargeDependentAttributeGenerator
from attributes.generators.composition.ElementalPropertyAttributeGenerator \
    import ElementalPropertyAttributeGenerator
from attributes.generators.composition.ElementFractionAttributeGenerator \
    import ElementFractionAttributeGenerator
from attributes.generators.composition.ElementPairPropertyAttributeGenerator \
    import ElementPairPropertyAttributeGenerator


from attributes.generators.composition.GCLPAttributeGenerator import \
    GCLPAttributeGenerator
from attributes.generators.composition\
    .IonicCompoundProximityAttributeGenerator import \
    IonicCompoundProximityAttributeGenerator
from attributes.generators.composition.IonicityAttributeGenerator import \
    IonicityAttributeGenerator
from attributes.generators.composition.MeredigAttributeGenerator import \
    MeredigAttributeGenerator
from attributes.generators.composition.StoichiometricAttributeGenerator \
    import StoichiometricAttributeGenerator


from attributes.generators.composition.ValenceShellAttributeGenerator import \
    ValenceShellAttributeGenerator
from attributes.generators.composition.YangOmegaAttributeGenerator import \
    YangOmegaAttributeGenerator
from attributes.generators.crystal.APRDFAttributeGenerator import \
    APRDFAttributeGenerator
from attributes.generators.crystal.ChemicalOrderingAttributeGenerator import \
    ChemicalOrderingAttributeGenerator
from attributes.generators.crystal.CoordinationNumberAttributeGenerator \
    import CoordinationNumberAttributeGenerator


from attributes.generators.crystal.CoulombMatrixAttributeGenerator import \
    CoulombMatrixAttributeGenerator
from attributes.generators.crystal\
    .EffectiveCoordinationNumberAttributeGenerator import \
    EffectiveCoordinationNumberAttributeGenerator
from attributes.generators.crystal.LatticeSimilarityAttributeGenerator import\
    LatticeSimilarityAttributeGenerator
from attributes.generators.crystal.LocalPropertyDifferenceAttributeGenerator \
    import  LocalPropertyDifferenceAttributeGenerator
from attributes.generators.crystal.LocalPropertyVarianceAttributeGenerator \
    import LocalPropertyVarianceAttributeGenerator


from attributes.generators.crystal.PackingEfficiencyAttributeGenerator import\
    PackingEfficiencyAttributeGenerator
from attributes.generators.crystal.PRDFAttributeGenerator import \
    PRDFAttributeGenerator
from attributes.generators.crystal.StructuralHeterogeneityAttributeGenerator \
    import StructuralHeterogeneityAttributeGenerator

from data.materials.CompositionEntry import CompositionEntry
from data.materials.CrystalStructureEntry import CrystalStructureEntry

# __all__ = ["attributes", "data", "models", "test", "utility", "vassal"]
