import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_param=0.05, beta=1e-3, lambda_reg=1e-4):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_param = sparsity_param
        self.beta = beta
        self.lambda_reg = lambda_reg

    def forward(self, x):
        hidden = torch.sigmoid(self.encoder(x))
        output = torch.sigmoid(self.decoder(hidden))
        return hidden, output

    def kl_divergence(self, rho_hat):
        rho = self.sparsity_param
        return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

    def loss_function(self, x, x_hat, hidden):
        mse_loss = F.mse_loss(x_hat, x)
        rho_hat = torch.mean(hidden, dim=0)
        kl_loss = self.kl_divergence(rho_hat)
        reg_loss = self.lambda_reg * (torch.sum(self.encoder.weight ** 2) + torch.sum(self.decoder.weight ** 2))
        return mse_loss + self.beta * kl_loss + reg_loss

def train_autoencoder(model, data, num_epochs=50, learning_rate=0.001, batch_size=64):
    data_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            data_batch = batch[0]
            optimizer.zero_grad()
            hidden, output = model(data_batch)
            loss = model.loss_function(data_batch, output, hidden)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}')
    return model