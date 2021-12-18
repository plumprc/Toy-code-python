import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = MNIST('datasets', train=True, download=True, transform=transform)
test_dataset = MNIST('datasets', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

def plot(conv_out, train_marks, test_marks, name, epoch=20):
    last_conv_out = conv_out[(epoch-1) * 70000:epoch * 70000]
    plt.figure(figsize=(16, 7))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    plt.subplot(1, 2, 1)
    for label in range(10):
        plt.scatter(last_conv_out[:60000, 0][np.array(train_marks[epoch-1]) == label],
                    last_conv_out[:60000, 1][np.array(train_marks[epoch-1]) == label],
                    s=1, c=colors[label])

    lgnd = plt.legend(['0', '1', '2', '3', '4', '5', '6',
                       '7', '8', '9'], loc="lower right")
    for idx in range(10):
        lgnd.legendHandles[idx]._sizes = [30]
    plt.xlabel('Activation of the 1st neuron', fontsize=12)
    plt.ylabel('Activation of the 2nd neuron', fontsize=12)
    plt.grid('on')

    plt.subplot(1, 2, 2)
    for label in range(10):
        plt.scatter(last_conv_out[-10000:, 0][np.array(test_marks[epoch-1]) == label],
                    last_conv_out[-10000:, 1][np.array(test_marks[epoch-1]) == label],
                    s=1, c=colors[label])

    lgnd = plt.legend(['0', '1', '2', '3', '4', '5', '6',
                       '7', '8', '9'], loc="lower right")
    for idx in range(10):
        lgnd.legendHandles[idx]._sizes = [30]
    plt.xlabel('Activation of the 1st neuron', fontsize=12)
    plt.ylabel('Activation of the 2nd neuron', fontsize=12)
    plt.grid('on')
    plt.savefig(name)

class LeNet_plus(nn.Module):
    def __init__(self):
        super(LeNet_plus, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.act2 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.act3 = nn.PReLU()
        self.conv4 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.act4 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.act5 = nn.PReLU()
        self.conv6 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.act6 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0)

        self.fc1 = nn.Linear(3*3*128, 2)
        self.act7 = nn.PReLU()
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)

        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)

        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)

        # save features last FC layer
        feat = self.act7(self.fc1(x))
        x = F.log_softmax(self.fc2(feat))

        return feat, x

    
class CenterLoss_with_autograd(nn.Module):
    def __init__(self, num_classes, dim_feat):
        super(CenterLoss_with_autograd, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = dim_feat
        self.centers = nn.Parameter(torch.randn(num_classes, dim_feat))

    def forward(self, y, deep_feat):
        batch_size = deep_feat.size(0)
        chosen_centers = self.centers.index_select(dim=0, index=y)

        # 2-norm
        p_norm = deep_feat.dist(chosen_centers, p=2)
        loss = 1.0 / 2 * p_norm / batch_size
        return loss
    
    # def backward(self, y, deep_feat):
    #     chosen_centers = self.centers.index_select(dim=0, index=y)
    #     return outputs.cpu() - indexing_centers


class CenterLoss_with_delta(nn.Module):
    def __init__(self, num_classes, dim_feat):
        super(CenterLoss_with_delta, self).__init__()
        self.num_classes = num_classes
        self.dim_feat = dim_feat
        self.centers = nn.Parameter(torch.randn(num_classes, dim_feat))

    def forward(self, y, deep_feat):
        device = torch.device('cuda:2')
        # Get loss
        batch_size = deep_feat.size(0)
        chosen_centers = self.centers.index_select(0, y.long())

        # Eq. 2
        p_norm = deep_feat.dist(chosen_centers, p=2)
        loss = 1.0 / 2 * p_norm / batch_size

        # Update centroids
        new_centers = torch.Tensor().to(device)
        # for each mark get new batch cantroide
        for i in range(self.num_classes):
            if i in y.data:
                # marks idx for current class
                idx_true = (y.data == i).nonzero().view(-1)

                # new centroid
                tmp = (deep_feat.index_select(0, idx_true)).mean(0).view(1, -1)
                new_centers = torch.cat((new_centers, tmp), 0)
            else:
                # zeros if no mark in batch
                new_centers = torch.cat(
                    (new_centers, torch.randn(1, self.dim_feat).to(device)), 0)

        # class buckets for batch
        hist = torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0, max=self.num_classes).to(device)

        # Eq. 4
        coeff = (hist / (1 + hist)).view(-1, 1)
        centers_grad = coeff * (self.centers - new_centers)

        return loss, centers_grad
    
    # def backward(self, y, deep_feat):
    #     chosen_centers = self.centers.index_select(dim=0, index=y)
    #     return outputs.cpu() - indexing_centers

    
def train(network, device, center_loss, optimizer, center_optimizer, epochs, lambda_, use_delta):
    global test_marks, train_marks

    nll_loss = nn.NLLLoss()

    train_loss_epochs = []
    test_loss_epochs = []
    train_accuracy_epochs = []
    test_accuracy_epochs = []

    try:
        for epoch in range(1, epochs+1):
            losses = []
            accuracies = []
            batch_marks = []

            network.train()
            for batch_idx, (X, y) in enumerate(train_loader):
                batch_marks.extend(y.numpy())
                X, y = X.to(device), y.to(device)
                network.zero_grad()
                center_loss.zero_grad()

                deep_feat, prediction = network(X)
                if use_delta:
                    # manually computed centroids
                    loss_center, grad = center_loss(y, deep_feat)
                    loss_batch = nll_loss(prediction, y) + \
                        lambda_ * loss_center

                    losses.append(loss_batch.item())

                    loss_batch.backward()

                    center_loss.zero_grad()
                    center_loss.centers.backward(grad)

                    optimizer.step()

                else:
                    # autograd
                    loss_batch = nll_loss(prediction, y) + \
                        lambda_ * center_loss(y, deep_feat)

                    losses.append(loss_batch.item())

                    loss_batch.backward()

                    optimizer.step()
                    center_optimizer.step()

                accuracies.append(
                    (np.argmax(prediction.cpu().data.numpy(), 1) == y.cpu().data.numpy()).mean())

                if batch_idx % 10 == 0:
                    sys.stdout.write('\rTrain Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, epochs, 
                        batch_idx * len(X), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), losses[-1]))

            train_marks.append(batch_marks)
            train_loss_epochs.append(np.mean(losses))
            train_accuracy_epochs.append(np.mean(accuracies))

            losses = []
            accuracies = []
            batch_marks = []

            network.eval()
            for X, y in test_loader:
                batch_marks.extend(y.numpy())
                X, y = X.to(device), y.to(device)
                deep_feat, prediction = network(X)

                if use_delta:
                    loss_center, params_grad = center_loss(y, deep_feat)
                    loss_batch = nll_loss(prediction, y) + \
                        lambda_ * loss_center
                else:
                    loss_batch = nll_loss(prediction, y) + \
                        lambda_ * center_loss(y, deep_feat)

                losses.append(loss_batch.item())
                accuracies.append(
                    (np.argmax(prediction.cpu().data.numpy(), 1) == y.cpu().data.numpy()).mean())

            test_marks.append(batch_marks)
            test_loss_epochs.append(np.mean(losses))
            test_accuracy_epochs.append(np.mean(accuracies))
            print('\nEpoch {0}. (Train/Test) Average loss: {1:.4f}/{2:.4f}\tAccuracy: {3:.4f}/{4:.4f}\n'.format(
                epoch, train_loss_epochs[-1], test_loss_epochs[-1],
                train_accuracy_epochs[-1], test_accuracy_epochs[-1]))
            
    except KeyboardInterrupt:
        pass

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_epochs, label='Train')
    plt.plot(test_loss_epochs, label='Test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.grid('on')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_epochs, label='Train accuracy')
    plt.plot(test_accuracy_epochs, label='Test accuracy')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.ylim(0.9, 1.0)
    plt.legend(loc=0, fontsize=16)
    plt.grid('on')
    plt.savefig('training_curve.png')

def get_out(self, input, output):
    global conv_out
    conv_out = np.concatenate((conv_out, output.cpu().data.numpy()), axis=0)

def init_net_and_train(epochs=20, lambda_=0, use_delta=0):
    device = torch.device("cuda:2")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = LeNet_plus().to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-4)

    if use_delta:
        centerloss = CenterLoss_with_delta(10, 2).to(device)
        center_optimizer = None
    else:
        centerloss = CenterLoss_with_autograd(10, 2).to(device)
        center_optimizer = optim.SGD(centerloss.parameters(), lr=0.5)

    network.fc1.register_forward_hook(get_out)

    train(network, device, centerloss, optimizer,
          center_optimizer, epochs, lambda_, use_delta)

epochs = 20
conv_out = np.empty([0,2])
test_marks = []
train_marks = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_', type=float, default=0, help='lambda_')
    parser.add_argument('--delta', type=float, default=0, help='delta')
    parser.add_argument('--name', type=str, default='feature', help='features')
    args = parser.parse_args()
    init_net_and_train(epochs, lambda_=args.lambda_, use_delta=args.delta)
    plot(conv_out, train_marks, test_marks, args.name, epochs)
