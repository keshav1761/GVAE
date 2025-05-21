#%%
import pandas as pd
DATA_PATH = r"C:\Users\kesha\Desktop\GVAE\data\raw\PI_DATA.csv"
data = pd.read_csv(DATA_PATH)
data.head()

print(data.shape)
print(data["Label"].value_counts())
#%%
import pandas as pd
from rdkit import Chem 
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return "not_implemented.pt"
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                mol_obj = Chem.MolFromSmiles(mol["Smiles"])
                if mol_obj is None:
                    print(f"Error: Invalid SMILES at index {index}: '{mol['Smiles']}'")
                    continue

                # Get node features
                node_feats = self._get_node_features(mol_obj)
                # Get edge features
                edge_feats = self._get_edge_features(mol_obj)
                # Get adjacency info
                edge_index = self._get_adjacency_info(mol_obj)
                # Get labels info
                labels = self._get_labels(mol["Label"])

                # Create data object
                data = Data(x=node_feats,
                            edge_index=edge_index,
                            edge_attr=edge_feats,
                            y=labels,
                            smiles=mol["Smiles"]
                            )
                
                
                if self.test:
                    torch.save(data,
                        os.path.join(self.processed_dir,
                                     f'data_test_{index}.pt'))
                else:
                    torch.save(data,
                        os.path.join(self.processed_dir,
                                     f'data_{index}.pt'))
            except Exception as e:
                print(f"Error processing molecule at index {index}: {e}")
                # Handle the error (e.g., skip this molecule or replace with default values)

    def _get_node_features(self, mol):
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())
            

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices
    
    def _get_labels(self, labels):
        labels = np.asarray([labels])
        return torch.tensor(labels, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                 f'data_{idx}.pt'))
        return data
    


#Dataset = MoleculeDataset(root= r"C:\Users\kesha\Desktop\GVAE\data" , filename="PI_DATA_train.csv")

#%%
print(Dataset[0].x)
print(Dataset[0].edge_attr)
print(Dataset[0].edge_index.t())
print(Dataset[0].y)

