import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

batch = 128
epochs = 30
device = 'cpu'

class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp

        return z
    
    def forward_loss(self, recon_x, x, mu, logvar):
        # BCE with sigmoid can be replaced by F.mse_loss()
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())

        return BCE + KLD

    def forward(self, x):
        # x: [batch, 784]
        h = self.encoder(x)
        # encoder: [batch, 40]
        mu, logvar = torch.chunk(h, 2, dim=1)
        # paras, z: [batch, 20]
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, self.forward_loss(recon_x, x, mu, logvar)


if __name__ == '__main__':
    dataset = datasets.MNIST(root='datasets', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for idx, (images, _) in enumerate(data_loader):
            images = images.to(device)
            images = images.view(images.size(0), -1)
            recon_images, loss = vae(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if idx % 100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item() / batch))

    torch.save(vae, 'vae.pth')
    sample = torch.randn(128, 20).to(device)
    recon_x = vae.decoder(sample)
    save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), 'sample.png')
