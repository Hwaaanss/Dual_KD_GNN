from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import (
    AllChem,
    ChemicalFeatures,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    rdReducedGraphs,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, Dataset


RDLogger.DisableLog("rdApp.warning")


class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "subgraph_edge_index":
            return int(self.subgraph_x.size(0))
        if key == "assign_index":
            return value.new_tensor([[self.x.size(0)], [self.subgraph_x.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


ATOM_SYMBOLS_100 = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
]
HYBRIDIZATIONS_127 = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_STEREOS_12 = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
MLFGNN_ATOMS_16 = ["C", "N", "O", "F", "Si", "Cl", "As", "Se", "Br", "Te", "I", "At"]
MLFGNN_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES_4 = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "P": 1.80,
}

FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)
ACIDIC_SMARTS = Chem.MolFromSmarts("[$([C,S,P](=O)[O;H,-1])]")
BASIC_SMARTS = Chem.MolFromSmarts(
    "[#7;+,$([N;H2&+0][C,c]),$([N;H1&+0]([C,c])[C,c]),$([N;H0&+0]([C,c])([C,c])[C,c])]"
)
DESCRIPTOR_FUNCS = [(name, fn) for name, fn in Descriptors._descList[:200]]
MORGAN_GENERATOR_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def one_hot_with_unknown(value, choices):
    return [1.0 if value == choice else 0.0 for choice in choices] + [0.0 if value in choices else 1.0]


def one_hot_no_unknown(value, choices):
    return [1.0 if value == choice else 0.0 for choice in choices]


def bitvect_to_array(fp, n_bits: int) -> np.ndarray:
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def get_auxiliary_atom_sets(mol):
    donor_atoms, acceptor_atoms = set(), set()
    for feature in FEATURE_FACTORY.GetFeaturesForMol(mol):
        if feature.GetFamily() == "Donor":
            donor_atoms.update(feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptor_atoms.update(feature.GetAtomIds())

    acidic_atoms = set(
        atom_idx
        for match in mol.GetSubstructMatches(ACIDIC_SMARTS)
        for atom_idx in match
    ) if ACIDIC_SMARTS is not None else set()
    basic_atoms = set(
        atom_idx
        for match in mol.GetSubstructMatches(BASIC_SMARTS)
        for atom_idx in match
    ) if BASIC_SMARTS is not None else set()
    return donor_atoms, acceptor_atoms, acidic_atoms, basic_atoms


def get_atom_features_127(atom) -> np.ndarray:
    features = []
    features.extend(one_hot_no_unknown(atom.GetSymbol(), ATOM_SYMBOLS_100))
    features.extend(one_hot_no_unknown(min(atom.GetDegree(), 5), list(range(6))))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetNumRadicalElectrons()))
    features.extend(one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS_127))
    features.extend(one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS))
    features.extend(one_hot_no_unknown(min(atom.GetTotalNumHs(), 4), list(range(5))))
    features.append(float(atom.IsInRing()))
    features.append(float(atom.GetIsAromatic()))
    return np.asarray(features, dtype=np.float32)


def get_bond_features_12(bond) -> np.ndarray:
    features = []
    features.extend(one_hot_no_unknown(bond.GetBondType(), BOND_TYPES_4))
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    features.extend(one_hot_no_unknown(bond.GetStereo(), BOND_STEREOS_12))
    return np.asarray(features, dtype=np.float32)


def get_atom_features_54(atom, donor_atoms, acceptor_atoms, acidic_atoms, basic_atoms) -> np.ndarray:
    features = []
    features.extend(one_hot_with_unknown(atom.GetSymbol(), MLFGNN_ATOMS_16))
    features.extend(one_hot_no_unknown(min(atom.GetDegree(), 5), list(range(6))))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetNumRadicalElectrons()))
    features.extend(one_hot_with_unknown(atom.GetHybridization(), MLFGNN_HYBRIDIZATIONS))
    features.append(float(atom.GetIsAromatic()))
    features.extend(one_hot_no_unknown(min(atom.GetTotalNumHs(), 4), list(range(5))))
    features.extend(one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS))
    features.append(float(atom.IsInRing()))
    features.extend([float(atom.IsInRingSize(size)) for size in [3, 4, 5, 6]])
    features.append(float(atom.GetMass() / 200.0))
    implicit_valence = min(max(int(atom.GetValence(Chem.ValenceType.IMPLICIT)), 0), 6)
    features.extend(one_hot_no_unknown(implicit_valence, list(range(7))))
    atom_idx = atom.GetIdx()
    features.append(float(atom_idx in acceptor_atoms))
    features.append(float(atom_idx in donor_atoms))
    features.append(float(atom_idx in acidic_atoms))
    features.append(float(atom_idx in basic_atoms))
    return np.asarray(features, dtype=np.float32)


def get_bond_features_13(bond) -> np.ndarray:
    features = [1.0]
    features.extend(one_hot_no_unknown(bond.GetBondType(), BOND_TYPES_4))
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    stereo_map = list(range(6))
    stereo_value = int(bond.GetStereo())
    stereo_value = stereo_value if stereo_value in stereo_map else 0
    features.extend(one_hot_no_unknown(stereo_value, stereo_map))
    return np.asarray(features, dtype=np.float32)


def get_rdkit_descriptor_200(mol) -> np.ndarray:
    values: list[float] = []
    for _, descriptor_fn in DESCRIPTOR_FUNCS:
        try:
            value = float(descriptor_fn(mol))
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            value = float(np.clip(value, -1e6, 1e6))
        except Exception:
            value = 0.0
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def fit_rdkit_descriptor_scaler(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            descriptors.append(np.zeros((200,), dtype=np.float32))
        else:
            descriptors.append(get_rdkit_descriptor_200(mol))

    descriptor_array = np.asarray(descriptors, dtype=np.float32)
    mean = descriptor_array.mean(axis=0)
    scale = descriptor_array.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def get_pubchem_like_fp(mol, n_bits: int = 881) -> np.ndarray:
    return bitvect_to_array(Chem.PatternFingerprint(mol, fpSize=n_bits), n_bits)


def get_erg_fp_441(mol) -> np.ndarray:
    fp = np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)
    if fp.shape[0] >= 441:
        return fp[:441]
    return np.pad(fp, (0, 441 - fp.shape[0]), mode="constant").astype(np.float32)


def build_subgraph_data(mol):
    cliques: list[list[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        cliques.append(sorted(set(ring)))
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))

    covered = {atom_idx for clique in cliques for atom_idx in clique}
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in covered:
            cliques.append([atom.GetIdx()])

    if not cliques:
        cliques = [[0]]

    sub_features = []
    assign_edges = []
    for subgraph_idx, clique in enumerate(cliques):
        is_ring = 1.0 if len(clique) > 2 else 0.0
        is_bond = 1.0 if len(clique) == 2 else 0.0
        is_singleton = 1.0 if len(clique) == 1 else 0.0
        sub_features.append([is_ring, is_bond, is_singleton, len(clique) / 8.0])
        for atom_idx in clique:
            assign_edges.append([atom_idx, subgraph_idx])

    sub_edges: list[list[int]] = []
    for left_idx in range(len(cliques)):
        clique_left = set(cliques[left_idx])
        for right_idx in range(left_idx + 1, len(cliques)):
            if clique_left.intersection(cliques[right_idx]):
                sub_edges.extend([[left_idx, right_idx], [right_idx, left_idx]])

    if sub_edges:
        sub_edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()
    else:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)

    assign_index = torch.tensor(assign_edges, dtype=torch.long).t().contiguous()
    sub_x = torch.tensor(np.asarray(sub_features, dtype=np.float32), dtype=torch.float)
    sub_batch = torch.zeros(sub_x.size(0), dtype=torch.long)
    return sub_x, sub_edge_index, assign_index, sub_batch


def smiles_to_graph_standard(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    donor_atoms, acceptor_atoms, acidic_atoms, basic_atoms = get_auxiliary_atom_sets(mol)
    x = torch.tensor(
        np.asarray([get_atom_features_127(atom) for atom in mol.GetAtoms()]),
        dtype=torch.float,
    )
    x_mlfgnn = torch.tensor(
        np.asarray(
            [
                get_atom_features_54(atom, donor_atoms, acceptor_atoms, acidic_atoms, basic_atoms)
                for atom in mol.GetAtoms()
            ]
        ),
        dtype=torch.float,
    )

    edge_pairs, edge_attr, edge_attr_mlfgnn = [], [], []
    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feature_12 = get_bond_features_12(bond)
        bond_feature_13 = get_bond_features_13(bond)
        edge_pairs.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        edge_attr.extend([bond_feature_12, bond_feature_12])
        edge_attr_mlfgnn.extend([bond_feature_13, bond_feature_13])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float)
        edge_attr_mlfgnn_tensor = torch.tensor(
            np.asarray(edge_attr_mlfgnn, dtype=np.float32),
            dtype=torch.float,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 12), dtype=torch.float)
        edge_attr_mlfgnn_tensor = torch.empty((0, 13), dtype=torch.float)

    pubchem_like = get_pubchem_like_fp(mol)
    fp_fpgnn = np.concatenate(
        [
            pubchem_like,
            bitvect_to_array(MACCSkeys.GenMACCSKeys(mol), 167),
            get_erg_fp_441(mol),
        ],
        axis=0,
    ).astype(np.float32)
    fp_mlfgnn = np.concatenate(
        [
            bitvect_to_array(MORGAN_GENERATOR_1024.GetFingerprint(mol), 1024),
            pubchem_like,
            get_erg_fp_441(mol),
        ],
        axis=0,
    ).astype(np.float32)
    sub_x, subgraph_edge_index, assign_index, subgraph_batch = build_subgraph_data(mol)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr_tensor,
        "x_mlfgnn": x_mlfgnn,
        "edge_attr_mlfgnn": edge_attr_mlfgnn_tensor,
        "fp_fpgnn": torch.tensor(fp_fpgnn, dtype=torch.float).unsqueeze(0),
        "fp_mlfgnn": torch.tensor(fp_mlfgnn, dtype=torch.float).unsqueeze(0),
        "rdkit_desc": torch.tensor(get_rdkit_descriptor_200(mol), dtype=torch.float).unsqueeze(0),
        "subgraph_x": sub_x,
        "subgraph_edge_index": subgraph_edge_index,
        "assign_index": assign_index,
        "subgraph_batch": subgraph_batch,
    }


def get_chemical_features(atom) -> np.ndarray:
    features = [atom.GetAtomicNum() / 100.0, atom.GetFormalCharge(), 1.0 if atom.GetIsAromatic() else 0.0]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    atom_hybridization = atom.GetHybridization()
    features.extend([1.0 if atom_hybridization == hybridization else 0.0 for hybridization in hybridizations])
    features.append(1.0 if atom_hybridization not in hybridizations else 0.0)
    atom_types = ["C", "N", "O", "S", "F", "Cl", "Br", "I"]
    atom_symbol = atom.GetSymbol()
    features.extend([1.0 if atom_symbol == atom_type else 0.0 for atom_type in atom_types])
    features.append(1.0 if atom_symbol not in atom_types else 0.0)
    features.append(float(atom.GetDegree()))
    return np.asarray(features, dtype=np.float32)


def get_physical_features(atom, conformer, center_of_mass) -> np.ndarray:
    features = []
    if conformer is not None and center_of_mass is not None:
        position = conformer.GetAtomPosition(atom.GetIdx())
        features.extend(
            [
                position.x - center_of_mass[0],
                position.y - center_of_mass[1],
                position.z - center_of_mass[2],
            ]
        )
    else:
        features.extend([0.0, 0.0, 0.0])
    features.append(atom.GetMass() / 100.0)
    features.append(VDW_RADII.get(atom.GetSymbol(), 1.70))
    return np.asarray(features, dtype=np.float32)


def get_bond_features_dual(bond, conformer) -> np.ndarray:
    features = []
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type = bond.GetBondType()
    features.extend([1.0 if bond_type == candidate else 0.0 for candidate in bond_types])
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    if conformer is not None:
        try:
            begin_pos = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
            end_pos = conformer.GetAtomPosition(bond.GetEndAtomIdx())
            features.append(float(begin_pos.Distance(end_pos)))
        except Exception:
            features.append(1.5)
    else:
        features.append(1.5)
    return np.asarray(features, dtype=np.float32)


def smiles_to_graph_dual(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        conformer = mol.GetConformer()
    except Exception:
        conformer = None

    mol = Chem.RemoveHs(mol)
    center_of_mass = None
    if conformer is not None:
        positions, masses = [], []
        for atom in mol.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            masses.append(atom.GetMass())
        center_of_mass = np.average(np.asarray(positions), axis=0, weights=np.asarray(masses))

    x_chem = torch.tensor(
        np.asarray([get_chemical_features(atom) for atom in mol.GetAtoms()]),
        dtype=torch.float,
    )
    x_phys = torch.tensor(
        np.asarray([get_physical_features(atom, conformer, center_of_mass) for atom in mol.GetAtoms()]),
        dtype=torch.float,
    )

    edge_index_rows, edge_features = [], []
    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feature = get_bond_features_dual(bond, conformer)
        edge_index_rows.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        edge_features.extend([bond_feature, bond_feature])

    if not edge_index_rows:
        return (
            x_chem,
            x_phys,
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 7), dtype=torch.float),
        )

    return (
        x_chem,
        x_phys,
        torch.tensor(edge_index_rows, dtype=torch.long).t().contiguous(),
        torch.tensor(np.asarray(edge_features), dtype=torch.float),
    )


def generate_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def scaffold_split(
    dataframe: pd.DataFrame,
    smiles_column: str = "smiles",
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = 42,
):
    np.random.seed(seed)
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for idx, smiles in enumerate(dataframe[smiles_column]):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold if scaffold else f"invalid_{idx}"].append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    train_indices, val_indices, test_indices = [], [], []
    train_cutoff = int(train_size * len(dataframe))
    val_cutoff = int((train_size + val_size) * len(dataframe))

    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_cutoff:
            train_indices.extend(scaffold_set)
        elif len(train_indices) + len(val_indices) + len(scaffold_set) <= val_cutoff:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    if not val_indices and len(scaffold_sets) > 1:
        midpoint = len(test_indices) // 2
        val_indices = test_indices[:midpoint]
        test_indices = test_indices[midpoint:]

    return train_indices, val_indices, test_indices


class Tox21Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        target_columns: list[str],
        indices=None,
        rdkit_desc_mean: np.ndarray | None = None,
        rdkit_desc_scale: np.ndarray | None = None,
    ):
        super().__init__()
        dataframe = pd.read_csv(data_path)
        if indices is not None:
            dataframe = dataframe.iloc[indices].reset_index(drop=True)
        self.smiles = dataframe["smiles"].tolist()
        labels = torch.tensor(dataframe[target_columns].values, dtype=torch.float)
        self.labels = torch.where(torch.isnan(labels), torch.tensor(-1.0), labels)
        self.rdkit_desc_mean = rdkit_desc_mean
        self.rdkit_desc_scale = rdkit_desc_scale

    def len(self):
        return len(self.smiles)

    def get(self, idx: int):
        graph_data = smiles_to_graph_standard(self.smiles[idx])
        rdkit_desc = torch.zeros((1, 200), dtype=torch.float)
        if graph_data is None:
            pass
        else:
            rdkit_desc = graph_data["rdkit_desc"].clone()
            if self.rdkit_desc_mean is not None and self.rdkit_desc_scale is not None:
                desc_array = rdkit_desc.squeeze(0).numpy()
                desc_array = (desc_array - self.rdkit_desc_mean) / self.rdkit_desc_scale
                rdkit_desc = torch.from_numpy(desc_array.astype(np.float32)).unsqueeze(0)

        if graph_data is None:
            return MolData(
                x=torch.zeros((1, 127), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 12), dtype=torch.float),
                x_mlfgnn=torch.zeros((1, 54), dtype=torch.float),
                edge_attr_mlfgnn=torch.empty((0, 13), dtype=torch.float),
                fp_fpgnn=torch.zeros((1, 1489), dtype=torch.float),
                fp_mlfgnn=torch.zeros((1, 2346), dtype=torch.float),
                rdkit_desc=rdkit_desc,
                subgraph_x=torch.tensor([[0.0, 0.0, 1.0, 0.125]], dtype=torch.float),
                subgraph_edge_index=torch.empty((2, 0), dtype=torch.long),
                assign_index=torch.tensor([[0], [0]], dtype=torch.long),
                subgraph_batch=torch.zeros(1, dtype=torch.long),
                y=self.labels[idx].clone(),
            )

        return MolData(
            x=graph_data["x"].clone(),
            edge_index=graph_data["edge_index"].clone(),
            edge_attr=graph_data["edge_attr"].clone(),
            x_mlfgnn=graph_data["x_mlfgnn"].clone(),
            edge_attr_mlfgnn=graph_data["edge_attr_mlfgnn"].clone(),
            fp_fpgnn=graph_data["fp_fpgnn"].clone(),
            fp_mlfgnn=graph_data["fp_mlfgnn"].clone(),
            rdkit_desc=rdkit_desc,
            subgraph_x=graph_data["subgraph_x"].clone(),
            subgraph_edge_index=graph_data["subgraph_edge_index"].clone(),
            assign_index=graph_data["assign_index"].clone(),
            subgraph_batch=graph_data["subgraph_batch"].clone(),
            y=self.labels[idx].clone(),
        )


class Tox21DualDataset(Dataset):
    def __init__(self, data_path: str, target_columns: list[str], indices=None):
        super().__init__()
        dataframe = pd.read_csv(data_path)
        if indices is not None:
            dataframe = dataframe.iloc[indices].reset_index(drop=True)
        self.smiles = dataframe["smiles"].tolist()
        labels = torch.tensor(dataframe[target_columns].values, dtype=torch.float)
        self.labels = torch.where(torch.isnan(labels), torch.tensor(-1.0), labels)

    def len(self):
        return len(self.smiles)

    def get(self, idx: int):
        graph_data = smiles_to_graph_dual(self.smiles[idx])
        if graph_data is None:
            x_chem = torch.zeros((1, 19), dtype=torch.float)
            x_phys = torch.zeros((1, 5), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 7), dtype=torch.float)
        else:
            x_chem, x_phys, edge_index, edge_attr = graph_data
            x_chem = x_chem.clone()
            x_phys = x_phys.clone()
            edge_index = edge_index.clone()
            edge_attr = edge_attr.clone()

        return Data(
            x=x_chem,
            x_chem=x_chem,
            x_phys=x_phys,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=self.labels[idx].clone(),
        )


def create_datasets(
    data_path: str,
    target_columns: list[str],
    seed: int = 42,
    dual: bool = False,
):
    dataframe = pd.read_csv(data_path)
    train_indices, val_indices, test_indices = scaffold_split(dataframe, seed=seed)
    dataset_cls = Tox21DualDataset if dual else Tox21Dataset
    rdkit_desc_mean = None
    rdkit_desc_scale = None
    if not dual:
        train_smiles = dataframe.iloc[train_indices]["smiles"].tolist()
        rdkit_desc_mean, rdkit_desc_scale = fit_rdkit_descriptor_scaler(train_smiles)
    print(
        f"Split for {'dual' if dual else 'standard'} features: "
        f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )
    if dual:
        return (
            dataset_cls(data_path, target_columns, train_indices),
            dataset_cls(data_path, target_columns, val_indices),
            dataset_cls(data_path, target_columns, test_indices),
        )
    return (
        dataset_cls(data_path, target_columns, train_indices, rdkit_desc_mean, rdkit_desc_scale),
        dataset_cls(data_path, target_columns, val_indices, rdkit_desc_mean, rdkit_desc_scale),
        dataset_cls(data_path, target_columns, test_indices, rdkit_desc_mean, rdkit_desc_scale),
    )
