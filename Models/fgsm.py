import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
import numpy as np
from matplotlib import pyplot as plt

batch = 128
epochs = 0
device = 'cpu'

class Classifier(nn.Module):
    def __init__(self, image_size, h_dim, cls_num):
        super(Classifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h_dim, cls_num)
        )

    def forward(self, x):
        # x: [batch, 784]

        return self.mlp(x)


if __name__ == '__main__':
    dataset = datasets.MNIST(root='datasets', train=True, transform=transforms.ToTensor(), download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    test_set = datasets.MNIST(root='datasets', train=False, transform=transforms.ToTensor(), download=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch)
    cls = Classifier(784, 100, 10).to(device)
    optimizer = torch.optim.Adam(cls.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for idx, (images, label) in enumerate(data_loader):
            images = images.to(device).view(images.shape[0], -1)
            pred = cls(images)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if idx % 100 == 0:
                print("Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss.item() / batch))

    # torch.save(cls, 'model.pth')
    cls = torch.load('model.pth')
    
    cls.eval()
    num_correct, FGSM = 0, 0
    eps = 0.05
    for idx, (images, label) in enumerate(test_loader):
        images = images.to(device).view(images.shape[0], -1)
        images.requires_grad = True
        pred = cls(images)
        pred_cls = torch.argmax(pred, -1)

        cls.zero_grad()
        loss = loss_fn(pred, label)
        loss.backward()

        adv_image = images + eps * images.grad.sign()
        pred_adv = cls(adv_image)
        pred_adv_cls = torch.argmax(pred_adv, -1)
        num_correct += (label == pred_cls).sum().item()
        FGSM += (label == pred_adv_cls).sum().item()

    print('acc:', num_correct / len(test_loader.dataset))
    print('adv_acc:', FGSM / len(test_loader.dataset))

    # image_grid = utils.make_grid(images.reshape(images.shape[0], 1, 28, 28))
    # plt.imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))

    for i in range(8):
        plt.subplot(3, 8, i+1)
        plt.imshow(adv_image[:8].reshape(8, 28, 28, 1).detach().numpy()[i], cmap='gray')
        plt.axis('off')
        plt.title(str(pred_adv_cls[:8][i].item()))
        
        plt.subplot(3, 8, i+9)
        plt.imshow((adv_image - images)[:8].reshape(8, 28, 28, 1).detach().numpy()[i], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 8, i+17)
        plt.imshow(images[:8].reshape(8, 28, 28, 1).detach().numpy()[i], cmap='gray')
        plt.axis('off')
        plt.title(str(pred_cls[:8][i].item()))

    plt.show()
