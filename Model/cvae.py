import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

batch = 128
epochs = 0
# device = torch.device('cuda:2')
device = 'cpu'
dataset = datasets.MNIST(root='datasets', train=True, transform=transforms.ToTensor(), download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

dic = {0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

class CVAE(nn.Module):
    def __init__(self, image_size=784, h_dim=512, z_dim=20, context_dim=10):
        super(CVAE, self).__init__()
        # self.context_embed = nn.Linear(1, 20)
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
    
    def forward(self, x, context):
        # x: [batch, 1, 28, 28]
        # context = self.context_embed(context)
        h = self.encoder(torch.cat([x, context], 1))
        # encoder: [batch, 40]
        mu, logvar = torch.chunk(h, 2, dim=1)
        # paras, z: [batch, 20]
        z = self.reparameterize(mu, logvar)
        
        return self.decoder(torch.cat([z, context], 1)), mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    return BCE + KLD

if __name__ == '__main__':
    cvae = CVAE().to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for idx, (images, label) in enumerate(data_loader):
            images = images.to(device)
            label = label.numpy()
            label_ = []
            for idx in label:
                label_.append(dic[idx])

            label = torch.from_numpy(np.array(label_)).float().to(device)

            # label = torch.zeros(label.shape[0], label.shape[1]).scatter_(1, label, 1).float()
            images = images.view(images.size(0), -1)
            recon_images, mu, logvar = cvae(images, label)
            loss = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if idx % 100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item() / batch))

    # torch.save(cvae, 'cvae.pth')
    cvae = torch.load('cvae.pth')
    cvae.eval()
    sample = torch.randn(32, 20).to(device)
    label = torch.from_numpy(np.tile(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 32).reshape(32, 10)).float().to(device)
    recon_x = cvae.decoder(torch.cat([sample, label[:32]], 1))
    save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), 'sample.png')
