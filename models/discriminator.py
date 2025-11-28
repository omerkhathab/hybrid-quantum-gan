import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_atoms, num_features, hidden_size=64):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_features = num_features
        input_size = num_atoms * num_atoms + num_atoms * num_features  # adj + features

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, adj, features):
        # Flatten adjacency and features, then concatenate
        adj_flat = adj.view(adj.size(0), -1)
        feat_flat = features.view(features.size(0), -1)
        x = torch.cat([adj_flat, feat_flat], dim=1)
        return self.model(x)