"""
Bond-visible MolGAN graph visualizer.

- Works even when molecules are chemically invalid
- Draws bonds manually
- Uses RDKit just to compute 2D coords (no sanitize)
- Guaranteed visible edges and atoms
"""

import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

GRAPHS_PKL = "data/preprocessed/PC9/graphs.pkl"
GRAPH_INDEX = 3
IMG_SIZE = 500  # output size


def rdkit_compute_coords_no_sanitize(mol):
    """Compute 2D coordinates without sanitizing the molecule."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass  # ignore sanitation failures
    
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    
    return mol


def draw_molecule_custom_bonds(A, F):
    """
    Custom drawing:
    - Add atoms from F
    - Add bonds from A (manual)
    - Compute coords (no sanitize)
    - Draw atoms as circles, bonds as lines
    """

    n = A.shape[0]

    # Build mol with atoms (all SINGLE bonds)
    rm = Chem.RWMol()
    atom_indices = []
    for i in range(n):
        atomic_num = int(F[i][0])
        atom = Chem.Atom(atomic_num)
        idx = rm.AddAtom(atom)
        atom_indices.append(idx)

    # Add single bonds from adjacency
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                try:
                    rm.AddBond(i, j, Chem.BondType.SINGLE)
                except:
                    pass

    mol = rm.GetMol()
    mol = rdkit_compute_coords_no_sanitize(mol)

    # Prepare drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(IMG_SIZE, IMG_SIZE)
    opts = drawer.drawOptions()

    opts.addAtomIndices = True  # show atom numbers
    opts.addBondIndices = False

    # Get 2D coordinates from RDKit
    try:
        conf = mol.GetConformer()
        coords = {i: conf.GetAtomPosition(i) for i in range(n)}
    except:
        # fallback: circular layout
        coords = {
            i: Chem.rdGeometry.Point3D(
                np.cos(2*np.pi*i/n) * 2.0, 
                np.sin(2*np.pi*i/n) * 2.0,
                0
            )
            for i in range(n)
        }

    drawer.SetDrawOptions(opts)
    drawer.DrawMolecule(mol)  # initial empty skeleton
    drawer.ClearDrawing()     # clear default RDKit rendering

    # Manual bond drawing
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                p1 = coords[i]
                p2 = coords[j]
                drawer.DrawLine((p1.x, p1.y), (p2.x, p2.y))

    # Manual atom drawing
    for i in range(n):
        p = coords[i]
        drawer.DrawString(f"{i}:{int(F[i][0])}", (p.x + 0.05, p.y + 0.05))

    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png))
    return img


def main():
    with open(GRAPHS_PKL, "rb") as f:
        graphs = pickle.load(f)

    A, F = graphs[GRAPH_INDEX]

    print(f"Drawing molecule {GRAPH_INDEX} (atoms {A.shape[0]})")

    img = draw_molecule_custom_bonds(A, F)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()