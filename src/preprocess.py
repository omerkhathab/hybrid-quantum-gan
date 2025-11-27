"""
Dataset preprocessor aligned with HQ-Cycle-MolGAN (pp2.pdf).

This version targets the explicit dataset locations provided by the user:
    PC9 -> data/raw/Archive/pc9/PC9_data/PC9_data/XYZ
    QM9 -> data/raw/Archive/qm9
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, rdmolops
from tqdm import tqdm

PC9_RAW_DIR = Path("data/raw/Archive/pc9/PC9_data/PC9_data/XYZ")
QM9_RAW_DIR = Path("data/raw/Archive/qm9")
PREPROCESSED_DIR = Path("data/preprocessed")
SUPPORTED_EXTS = {".sdf", ".mol", ".xyz"}

# Silence RDKit's verbose warning/deprecation spam (GetValence, stereo ambiguity, etc.)
RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("warning")


# --- Feature extraction identical to MolGAN/HQ-MolGAN (paper pp2.pdf) ---
def atom_features(atom: Chem.Atom) -> np.ndarray:
    return np.array(
        [
            atom.GetAtomicNum(),  # atomic number
            atom.GetDegree(),
            atom.GetImplicitValence(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
        ],
        dtype=np.float32,
    )


def mol_to_graph(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    adjacency = rdmolops.GetAdjacencyMatrix(mol)
    features = np.array([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    return adjacency, features


def _scan_molecule_files(base_dir: Path) -> Iterable[Path]:
    for root, _, files in os.walk(base_dir):
        for file in files:
            path = Path(root) / file
            if path.suffix.lower() in SUPPORTED_EXTS:
                yield path


def _load_molecules_from_file(filepath: Path) -> List[Chem.Mol]:
    ext = filepath.suffix.lower()

    if ext == ".sdf":
        supplier = Chem.SDMolSupplier(str(filepath), removeHs=False)
        return [mol for mol in supplier if mol is not None]

    if ext == ".mol":
        mol = Chem.MolFromMolFile(str(filepath), sanitize=True, removeHs=False)
        return [mol] if mol is not None else []

    if ext == ".xyz":
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.rstrip() for line in f.readlines() if line.strip()]
        
        if len(lines) < 3:
            return []
        
        # Parse non-standard PC9/QM9 XYZ format:
        # Line 1: atom count
        # Line 2: metadata (skip)
        # Lines 3+: element x y z [extra columns...]
        try:
            num_atoms = int(lines[0].strip())
            if num_atoms <= 0 or num_atoms > 1000:  # sanity check
                return []
            
            # Get atom lines (skip line 0=count, line 1=metadata)
            atom_lines = lines[2:2+num_atoms]
            if len(atom_lines) < num_atoms:
                return []
            
            # Reconstruct standard XYZ block for RDKit
            xyz_block = f"{num_atoms}\n\n"
            valid_atoms = 0
            for line in atom_lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                element = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    xyz_block += f"{element:2s} {x:>15.10f} {y:>15.10f} {z:>15.10f}\n"
                    valid_atoms += 1
                except (ValueError, IndexError):
                    continue
            
            if valid_atoms == 0:
                return []
            
            mol = Chem.MolFromXYZBlock(xyz_block)
            if mol is None:
                return []
            
            # CRITICAL FIX: Detect bonds from atomic distances
            try:
                # Convert to editable molecule for bond addition
                rwmol = Chem.RWMol(mol)
                
                # Get 3D distance matrix
                dist_matrix = AllChem.Get3DDistanceMatrix(rwmol)
                
                # Define bond distance thresholds (approximate)
                bond_thresholds = {
                    ('C', 'C'): 1.7, ('C', 'H'): 1.1, ('C', 'N'): 1.5, ('C', 'O'): 1.4,
                    ('N', 'H'): 1.0, ('O', 'H'): 1.0, ('N', 'N'): 1.5, ('O', 'O'): 1.5,
                    ('C', 'F'): 1.4, ('C', 'Cl'): 1.8, ('C', 'Br'): 2.0, ('C', 'I'): 2.2,
                    ('H', 'H'): 1.8  # for very close hydrogens
                }
                
                # Get atomic symbols
                symbols = [atom.GetSymbol() for atom in rwmol.GetAtoms()]
                
                # Add bonds based on distances
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        dist = dist_matrix[i, j]
                        pair = tuple(sorted([symbols[i], symbols[j]]))
                        threshold = bond_thresholds.get(pair, 1.6)  # default to C-C distance
                        
                        if dist <= threshold:
                            rwmol.AddBond(i, j, Chem.BondType.SINGLE)
                
                # Convert back to regular molecule
                mol = rwmol.GetMol()
                
                # Sanitize to ensure proper valence
                Chem.SanitizeMol(mol)
                    
            except Exception as e:
                # If bond detection fails, try basic sanitization
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass  # Keep the molecule even if sanitization fails
            
            return [mol]
        except (ValueError, IndexError, AttributeError):
            return []

    return []


def _serialize_graphs(graphs: List[Tuple[np.ndarray, np.ndarray]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        pickle.dump(graphs, f)


def preprocess_dataset():
    datasets = {
        "PC9": PC9_RAW_DIR,
        "QM9": QM9_RAW_DIR,
    }

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, object]] = {}

    for dataset_name, base_dir in datasets.items():
        if not base_dir.exists():
            print(f"[{dataset_name}] Skipped - missing directory: {base_dir}")
            continue

        molecule_files = list(_scan_molecule_files(base_dir))
        if not molecule_files:
            print(f"[{dataset_name}] No supported molecule files under {base_dir}")
            continue

        graphs: List[Tuple[np.ndarray, np.ndarray]] = []
        sources: set[str] = set()

        for file_path in tqdm(
            molecule_files,
            desc=f"Parsing {dataset_name}",
            unit="file",
            dynamic_ncols=True,
        ):
            molecules = _load_molecules_from_file(file_path)
            for mol in molecules:
                adjacency, features = mol_to_graph(mol)
                graphs.append((adjacency, features))
                if len(graphs) == 4:  # Debug first few molecules
                    print(f"Debug molecule {len(graphs)-1}: {features.shape[0]} atoms, atomic nums: {features[:, 0].astype(int)}")
            sources.add(str(file_path))

        if not graphs:
            print(f"[{dataset_name}] No valid molecules parsed.")
            continue

        dataset_dir = PREPROCESSED_DIR / dataset_name
        out_file = dataset_dir / "graphs.pkl"
        _serialize_graphs(graphs, out_file)

        manifest[dataset_name] = {
            "graph_count": len(graphs),
            "source_count": len(sources),
            "sample_sources": sorted(sources)[:5],
            "pickle": str(out_file),
        }

        print(f"[{dataset_name}] graphs: {len(graphs)} -> {out_file}")

    manifest_path = PREPROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    preprocess_dataset()

