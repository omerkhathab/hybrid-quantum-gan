import pickle
import torch
import torch.nn as nn

# Load preprocessed graphs to determine sizes
with open("data/preprocessed/PC9/graphs.pkl", "rb") as f:
    graphs = pickle.load(f)
# graphs is a list of (adjacency, features) tuples

# Determine max number of atoms and feature size from data
num_atoms = max(adj.shape[0] for adj, _ in graphs)
num_features = graphs[0][1].shape[1]  # 5 for MolGAN-style

class Generator(nn.Module):
    def __init__(self, noise_dim=16, hidden_size=64):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_features = num_features
        output_size = self.num_atoms * self.num_atoms + self.num_atoms * self.num_features

        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.model(z)
        adj_flat = out[:, :self.num_atoms * self.num_atoms]
        feat_flat = out[:, self.num_atoms * self.num_atoms:]
        adj = adj_flat.view(-1, self.num_atoms, self.num_atoms)
        features = feat_flat.view(-1, self.num_atoms, self.num_features)
        return adj, features

# Example usage:
# generator = Generator()
# noise = torch.randn(batch_size, 16)
# adj, features = generator(noise)