"""
Script for testing the Dense and Convolutional Bayesian Layers using the MNIST dataset.
Currently used architecture: LeNet-5

@author: Konstantinos P. Panousis
Cyprus University of Technology
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from layers import ConvBayesian, DenseBayesian
from utils import parameterConstraints, model_kl_divergence_loss


class Net(nn.Module):
    """
    Create the network architecture using the custom layers.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvBayesian(1, 20, kernel_size = 5, stride = 1, padding = 0, competitors = 1,
                                  activation = 'lwta', prior_mean=0, prior_scale=1. ,  ibp = True)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = ConvBayesian(20, 50, kernel_size = 5, stride = 1, padding = 0, competitors = 1,
                                  activation = 'lwta', prior_mean=0, prior_scale=1.,  ibp = True )
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = DenseBayesian(input_features= 800 , output_features=500, competitors = 1, activation = 'lwta',
                   prior_mean=0, prior_scale=1. , ibp = True)
        self.fc2 = DenseBayesian(input_features= 500 , output_features=10, competitors = 1, activation = 'linear',
                   prior_mean=0, prior_scale=1. ,  ibp = True)


    def forward(self, x):
        """
        Override the forward function. Connect the layers.

        :param x: the input to the layers

        :return: the output logits
        """

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), 800)

        x = self.fc1(x)

        x = self.fc2(x)

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Function to train the model.

    :param args: obj: object with various parameters for training
    :param model: nn.Module: an instance of nn Module defining our model
    :param device: the device to use, e.g. cuda or gpu
    :param train_loader: torch Data Loader: containing the training data
    :param optimizer: torch optimizer: the optimizer for training
    :param epoch: int: The current epoch

    :return: null
    """

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        constraints = parameterConstraints()
        output = model(data)

        loss = model.loss(output, target)
        #kl_loss = model.kl_loss(model)

        #loss += kl_loss

        loss.backward()
        optimizer.step()
        model.apply(constraints)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, train_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += model.loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\n########################')
    print('Omitted Components:')
    shape = list(model.children())[0].t_pi.size(0)
    print('Layer 1:', (torch.sigmoid(list(model.children())[0].t_pi) < 1e-2).sum(), '/', shape)

    shape = list(model.children())[2].t_pi.size(0)
    print('Layer 2:', (torch.sigmoid(list(model.children())[2].t_pi) < 1e-2).sum(), '/', shape)

    shape = list(model.children())[4].t_pi.size(0) * list(model.children())[4].t_pi.size(1)
    print('Layer 3:', (torch.sigmoid(list(model.children())[4].t_pi) < 1e-2).sum(), '/', shape)
    print('########################\n')
    test_loss /= len(test_loader.dataset)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)


    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    model.loss = nn.CrossEntropyLoss()
    model.kl_loss = model_kl_divergence_loss
    model.kl_weight = 0.1

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, train_loader)
        scheduler.step()

    if args.save_model:
        print('AAAAAAAAAAAAAAAAAAAA')
        torch.save(model.state_dict(), "mnist_dense.pt")


if __name__ == '__main__':
    main()