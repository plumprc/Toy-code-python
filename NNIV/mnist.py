import torch
import torch.nn as nn
from torchvision import datasets, transforms

batch = 128
epochs = 3
device = 'cpu'

class MLP(nn.Module):
    def __init__(self, image_size, h_dim, cls_num):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h_dim, cls_num)
        )

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = x.view(x.shape[0], -1)

        return self.mlp(x)


class Conv(nn.Module):
    def __init__(self, in_channel, h_dim, cls_num):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channel, 16, 3, 3)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(h_dim, cls_num)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(x)

        return self.proj(x)


if __name__ == '__main__':
    dataset = datasets.MNIST(root='datasets', train=True, transform=transforms.ToTensor(), download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    test_set = datasets.MNIST(root='datasets', train=False, transform=transforms.ToTensor(), download=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch)
    cls = MLP(784, 100, 10).to(device)
    # cls = Conv(1, 16 * 81, 10).to(device)
    optimizer = torch.optim.Adam(cls.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    ''' TRAIN '''
    for epoch in range(epochs):
        for idx, (images, label) in enumerate(data_loader):
            images = images.to(device)
            pred = cls(images)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if idx % 100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item() / batch))

    # torch.save(cls, 'checkpoints/mlp.pth')
    # cls = torch.load('checkpoints/mlp.pth')
    
    ''' TEST '''
    cls.eval()
    num_correct = 0
    for idx, (images, label) in enumerate(test_loader):
        pred = cls(images)
        pred_cls = torch.argmax(pred, -1)
        num_correct += (label == pred_cls).sum().item()

    print('acc:', num_correct / len(test_loader.dataset))
