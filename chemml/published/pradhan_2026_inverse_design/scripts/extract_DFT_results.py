import os
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

# --- 1. SET YOUR PATHS ---
dft_folder = "../100k_den/DFT"
out_file = os.path.join(dft_folder, "champion_density_1st_gen.out")
xyz_file = os.path.join(dft_folder, "champion_density_1st_gen.xyz")

# --- 2. EXTRACT ORCA DATA ---
polarizability_bohr3 = None

# Search the output file for the isotropic polarizability
with open(out_file, "r") as f:
    for line in f:
        if "Isotropic polarizability" in line:
            # Grabs the number at the end of the line
            polarizability_bohr3 = float(line.split(":")[1].strip())
            break

if polarizability_bohr3 is None:
    print("Error: Polarizability not found! Make sure ORCA finished successfully.")
else:
    # Convert Bohr^3 to Angstroms^3
    alpha_A3 = polarizability_bohr3 * 0.148184 

    # --- 3. CALCULATE VOLUME & DENSITY ---
    # Load the optimized geometry
    mol = Chem.MolFromXYZFile(xyz_file)
    
    # RDKit loses bond info from XYZ files
    original_smiles = "Oc1nsnc1c1nsnc1O"
    mol_for_mass = Chem.MolFromSmiles(original_smiles)
    molar_mass = Descriptors.MolWt(mol_for_mass) # in g/mol

    # Calculate Volume
    v_vdw = AllChem.ComputeMolVolume(mol) # in A^3
    packing_fraction = 0.60
    v_bulk = v_vdw / packing_fraction # Bulk volume per molecule in A^3

    # Calculate Density (g/cm^3)
    # Conversion factor: 1 A^3 = 1e-24 cm^3. Avogadro's number = 6.022e23
    density_g_cm3 = (molar_mass / (v_bulk * 1e-24)) / 6.022e23
    
    # Convert Density to kg/m^3 for your paper
    density_kg_m3 = density_g_cm3 * 1000

    # --- 4. CALCULATE REFRACTIVE INDEX (Lorentz-Lorenz) ---
    # Number Density (molecules per A^3)
    N = 1 / v_bulk 
    
    # Lorentz-Lorenz term: (4 * pi / 3) * N * alpha
    LL_term = (4 * math.pi / 3) * N * alpha_A3
    
    # Solve for n: (n^2 - 1) / (n^2 + 2) = LL_term
    # n = sqrt((1 + 2*LL_term) / (1 - LL_term))
    RI = math.sqrt((1 + 2 * LL_term) / (1 - LL_term))

    # --- 5. PRINT THE RESULTS ---
    print(f"--- QUANTUM RESULTS FOR LATEX ---")
    print(f"Van der Waals Volume:     {v_vdw:.2f} Å³")
    print(f"Isotropic Polarizability: {alpha_A3:.2f} Å³")
    print(f"---------------------------------")
    print(f"Bulk Density: {density_kg_m3:.2f} kg/m³")