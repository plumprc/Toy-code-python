import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

batch = 128
epochs = 30
device = 'cpu'

dic = {0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

class CVAE(nn.Module):
    def __init__(self, image_size, h_dim, z_dim, context_dim):
        super(CVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(image_size + context_dim, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + context_dim, h_dim),
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

    def forward(self, x, context):
        # x: [batch, 784]
        h = self.encoder(torch.cat([x, context], 1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(torch.cat([z, context], 1))

        return recon_x, self.forward_loss(recon_x, x, mu, logvar)


if __name__ == '__main__':
    dataset = datasets.MNIST(root='datasets', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    cvae = CVAE(784, 400, 20, 10).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for idx, (images, label) in enumerate(data_loader):
            images = images.to(device)
            label = label.numpy()
            label_ = []
            for idx in label:
                label_.append(dic[idx])

            label = torch.from_numpy(np.array(label_)).float().to(device)
            images = images.view(images.size(0), -1)
            recon_images, loss= cvae(images, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if idx % 100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item() / batch))

    torch.save(cvae, 'cvae.pth')
    # cvae = torch.load('cvae.pth')
    cvae.eval()
    sample = torch.randn(32, 20).to(device)
    label = torch.from_numpy(np.tile(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 32).reshape(32, 10)).float().to(device)
    recon_x = cvae.decoder(torch.cat([sample, label[:32]], 1))
    save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), 'sample.png')
