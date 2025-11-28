import os
from datetime import datetime
import pickle
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

# Import Generator and Discriminator
from models.generator import Generator, num_atoms, num_features
from models.discriminator import Discriminator

# Custom Dataset for molecule graphs
class MoleculeGraphDataset(Dataset):
    def __init__(self, graph_file):
        with open(graph_file, "rb") as f:
            self.graphs = pickle.load(f)
        self.num_atoms = num_atoms
        self.num_features = num_features

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        adj, features = self.graphs[idx]
        # Pad to max size if needed
        pad_adj = torch.zeros(self.num_atoms, self.num_atoms, dtype=torch.float32)
        pad_feat = torch.zeros(self.num_atoms, self.num_features, dtype=torch.float32)
        n = adj.shape[0]
        pad_adj[:n, :n] = torch.tensor(adj, dtype=torch.float32)
        pad_feat[:n, :] = torch.tensor(features, dtype=torch.float32)
        return pad_adj, pad_feat

def train_gan(
    generator,
    discriminator,
    dataloader,
    log_dir_name,
    fixed_noise,
    num_epochs=100,
    device="cpu",
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir_name, run_id)
    writer = SummaryWriter(log_dir=log_dir)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        for i, (real_adj, real_feat) in enumerate(dataloader):
            real_adj = real_adj.to(device)
            real_feat = real_feat.to(device)
            batch_size = real_adj.size(0)

            # Train Discriminator with real data
            d_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            real_output = discriminator(real_adj, real_feat)
            d_real_loss = criterion(real_output, real_labels)
            d_real_loss.backward()

            # Train Discriminator with fake data
            z = torch.randn(batch_size, generator.model[0].in_features, device=device)
            fake_adj, fake_feat = generator(z)
            fake_output = discriminator(fake_adj.detach(), fake_feat.detach())
            d_fake_loss = criterion(fake_output, fake_labels)
            d_fake_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, generator.model[0].in_features, device=device)
            fake_adj, fake_feat = generator(z)
            fake_output = discriminator(fake_adj, fake_feat)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # Log losses to TensorBoard
            step = epoch * len(dataloader) + i
            writer.add_scalar("Generator Loss", g_loss.item(), step)
            writer.add_scalar("Discriminator Real Loss", d_real_loss.item(), step)
            writer.add_scalar("Discriminator Fake Loss", d_fake_loss.item(), step)

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                    f"Generator Loss: {g_loss.item():.4f}, "
                    f"Discriminator Real Loss: {d_real_loss.item():.4f}, "
                    f"Discriminator Fake Loss: {d_fake_loss.item():.4f}"
                )

    writer.close()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load dataset
    dataset = MoleculeGraphDataset("data/preprocessed/PC9/graphs.pkl")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate models
    generator = Generator().to(device)
    discriminator = Discriminator(num_atoms, num_features).to(device)

    # Fixed noise for monitoring
    fixed_noise = torch.randn(9, generator.model[0].in_features, device=device)

    train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        log_dir_name="logs",
        fixed_noise=fixed_noise,
        num_epochs=10,
        device=device,
    )

    # Save trained generator model
    torch.save(generator.state_dict(), "resources/generator_model.pth")