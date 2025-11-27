from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuration: which graph to visualize (0-based index)
GRAPH_INDEX = 3  # Change this to visualize different molecules
VISUALIZATION_MODE = "grid"  # Options: "single", "grid", "detailed", "3d_style"

# Load preprocessed graphs
print("Loading graphs...")
with open("data/preprocessed/PC9/graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

print(f"Total graphs available: {len(graphs)}")

def create_molecule_from_graph(A, F):
    """Create RDKit molecule from adjacency matrix and features"""
    num_atoms = A.shape[0]
    mol = Chem.RWMol()

    # Add atoms using atomic number from features
    for i in range(num_atoms):
        atomic_num = int(F[i][0])  # first feature = atomic number
        atom = Chem.Atom(atomic_num)
        mol.AddAtom(atom)

    # Add bonds based on adjacency matrix
    bonds_added = 0
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            if A[i, j] > 0:  # Check for any non-zero value indicating a bond
                mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
                bonds_added += 1

    return mol, bonds_added

def visualize_single_molecule(mol, graph_index, save_path=None):
    """Create a detailed single molecule visualization"""
    try:
        # Sanitize and compute 2D coordinates
        Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)

        # Calculate properties
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        print(f"\n=== Molecule {graph_index} Properties ===")
        print(f"Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
        print(f"Molecular Weight: {mol_weight:.2f} Da")
        print(f"LogP: {logp:.2f}")
        print(f"TPSA: {tpsa:.2f} Å²")
        print(f"H-bond donors: {hbd}, H-bond acceptors: {hba}")
        print(f"Rotatable bonds: {Descriptors.NumRotatableBonds(mol)}")
        print(f"SMILES: {Chem.MolToSmiles(mol)}")

        # Create enhanced visualization with atom indices
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
        drawer.drawOptions().addAtomIndices = True
        drawer.drawOptions().addBondIndices = False
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().continuousHighlight = False

        # Highlight different atom types
        highlight_atoms = {}
        highlight_colors = {}

        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 6:  # Carbon
                highlight_atoms[atom.GetIdx()] = (0.8, 0.8, 0.8)  # Light gray
            elif atomic_num == 7:  # Nitrogen
                highlight_atoms[atom.GetIdx()] = (0.0, 0.0, 1.0)  # Blue
            elif atomic_num == 8:  # Oxygen
                highlight_atoms[atom.GetIdx()] = (1.0, 0.0, 0.0)  # Red
            elif atomic_num == 9:  # Fluorine
                highlight_atoms[atom.GetIdx()] = (0.0, 1.0, 0.0)  # Green

        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        drawer.FinishDrawing()

        # Get the image
        img_data = drawer.GetDrawingText()
        with open('/tmp/mol.png', 'wb') as f:
            f.write(img_data)
        img = Image.open('/tmp/mol.png')

        if save_path:
            img.save(save_path)
            print(f"Saved visualization to: {save_path}")

        return img

    except Exception as e:
        print(f"Error in detailed visualization: {e}")
        # Fallback to basic visualization
        return Draw.MolToImage(mol, size=(400, 400))

def visualize_molecule_grid(graphs, start_idx=0, num_molecules=9, save_path=None):
    """Create a grid visualization of multiple molecules"""
    molecules = []
    legends = []

    for i in range(start_idx, min(start_idx + num_molecules, len(graphs))):
        A, F = graphs[i]
        mol, bonds_added = create_molecule_from_graph(A, F)

        try:
            Chem.SanitizeMol(mol)
            AllChem.Compute2DCoords(mol)
            molecules.append(mol)
            legends.append(f"Mol {i}\n{Chem.rdMolDescriptors.CalcMolFormula(mol)}")
        except Exception as e:
            print(f"Skipping molecule {i}: {e}")
            continue

    if not molecules:
        print("No valid molecules to visualize")
        return None

    # Create grid image
    img = Draw.MolsToGridImage(molecules, legends=legends,
                              molsPerRow=3, subImgSize=(300, 300))

    if save_path:
        img.save(save_path)
        print(f"Saved grid visualization to: {save_path}")

    return img

def visualize_3d_style(mol, save_path=None):
    """Create a 3D-style visualization with depth cues"""
    try:
        Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)

        # Create drawing with enhanced visual effects
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().bondLineWidth = 2
        drawer.drawOptions().addStereoAnnotation = True

        # Add some visual enhancements
        drawer.drawOptions().continuousHighlight = True

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        img_data = drawer.GetDrawingText()
        with open('/tmp/mol_3d.png', 'wb') as f:
            f.write(img_data)
        img = Image.open('/tmp/mol_3d.png')

        if save_path:
            img.save(save_path)
            print(f"Saved 3D-style visualization to: {save_path}")

        return img

    except Exception as e:
        print(f"Error in 3D visualization: {e}")
        return Draw.MolToImage(mol, size=(400, 400))

# Main visualization logic
if VISUALIZATION_MODE == "single":
    print(f"Visualizing single molecule {GRAPH_INDEX}...")
    A, F = graphs[GRAPH_INDEX]
    mol, bonds_added = create_molecule_from_graph(A, F)
    print(f"Added {bonds_added} bonds from adjacency matrix")

    img = visualize_single_molecule(mol, GRAPH_INDEX, save_path=f"molecule_{GRAPH_INDEX}_detailed.png")
    img.show()

elif VISUALIZATION_MODE == "grid":
    print("Creating molecule grid visualization...")
    img = visualize_molecule_grid(graphs, start_idx=0, num_molecules=9, save_path="molecule_grid.png")
    if img:
        img.show()

elif VISUALIZATION_MODE == "detailed":
    print(f"Creating detailed visualization for molecule {GRAPH_INDEX}...")
    A, F = graphs[GRAPH_INDEX]
    mol, bonds_added = create_molecule_from_graph(A, F)
    print(f"Added {bonds_added} bonds from adjacency matrix")

    img = visualize_single_molecule(mol, GRAPH_INDEX, save_path=f"molecule_{GRAPH_INDEX}_detailed.png")
    img.show()

elif VISUALIZATION_MODE == "3d_style":
    print(f"Creating 3D-style visualization for molecule {GRAPH_INDEX}...")
    A, F = graphs[GRAPH_INDEX]
    mol, bonds_added = create_molecule_from_graph(A, F)
    print(f"Added {bonds_added} bonds from adjacency matrix")

    img = visualize_3d_style(mol, save_path=f"molecule_{GRAPH_INDEX}_3d.png")
    img.show()

else:
    print(f"Unknown visualization mode: {VISUALIZATION_MODE}")
    print("Available modes: 'single', 'grid', 'detailed', '3d_style'")