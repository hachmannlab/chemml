# Reward Hacking in Materials Discovery: An Interpretable Inverse Design Framework and its Physical Limits

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the data, extraction scripts, and generative algorithms for the paper **"Reward Hacking in Materials Discovery: An Interpretable Inverse Design Framework and its Physical Limits"**. 

Our framework introduces an interpretable, feature-correlation-driven inverse engineering protocol that bridges robust statistical descriptor analysis with constrained Genetic Algorithms (GAs). Crucially, this repository provides the code to reproduce our diagnosis of the "extrapolation crisis" in materials informatics, demonstrating how 2D machine learning models can mathematically hallucinate topological correlations (reward hacking) that physically collapse under 3D steric constraints.

## Table of Contents
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage: Reproducing the Workflow](#usage-reproducing-the-workflow)
- [Citation](#citation)
- [License](#license)

---

## Repository Structure

```text
Inverse_Design_GA/
│
├── README.md                  <- You are here!
├── requirements.txt           <- Python dependencies for the environment
│
├── data/                      <- Raw datasets and building block fragments
│   ├── paper13_smiles_ri_mf.csv   # 1st-Gen Polyimide dataset (Refractive Index)
│   ├── 100k_den.csv           # 1st-Gen Organic dataset (Density) 
│   ├── paper9_MF.csv.zip              # Morgan Fingerprints for the Organic dataset - Zipped for GitHub
│   ├── den_bb.txt                     # Building blocks for Small Organics
│   └── ri_bb.txt                    # Building blocks for Polyimides
│
├── scripts/                   <- Executable Python scripts
│   ├── run_polyimide_ga.py    <- Trains the PI surrogate and runs the constrained GA
│   ├── run_density_ga.py      <- Trains the Density surrogate and runs the unconstrained GA
│   ├── extract_z_scores.py    <- Statistical extraction of structural motifs (Z-scores/A-scores)
│   └── dft_processing.py      <- Extracts isotropic polarizability and calculates true density/RI from ORCA outputs
│
└── outputs/                   <- All output files go here
    └── .gitkeep
```
---

## Citation

Please cite any use of this work as:
```text
Aatish Pradhan, Gaurav Vishwakarma, Johannes Hachmann. Reward Hacking in Materials Discovery: An Interpretable Inverse Design Framework and its Physical Limits. ChemRxiv. 24 March 2026.
DOI: https://doi.org/10.26434/chemrxiv.15001186/v1
```