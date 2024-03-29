from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from simple_conv_net_func import diff_mse
from simple_conv_net_func import conv2d_scalar, pool2d_scalar, relu_scalar, fc_layer_scalar
from simple_conv_net_func import conv2d_vector, pool2d_vector, relu_vector, fc_layer_vector


class SimpleConvNet(nn.Module):
    def __init__(self, device):
        super(SimpleConvNet, self).__init__()
        self.device = device
        self.conv_layer = nn.Conv2d(in_channels=1,
                                    out_channels=20, #change back to 20
                                    kernel_size=5,
                                    stride=1,
                                    padding=0,
                                    dilation=1,
                                    groups=1,
                                    bias=True)
        self.fc_layer1 = nn.Linear(in_features=20 * 12 * 12, out_features=500)
        self.fc_layer2 = nn.Linear(in_features=500, out_features=10)
        self.to(device)


    def forward(self, x):
        z_conv = self.conv_layer(x)
        my_conv = conv2d_scalar(x, conv_weight=self.conv_layer.weight, conv_bias=self.conv_layer.bias, device = self.device)
        
        print("TEST!!! conv = ", diff_mse(my_conv, z_conv))
        z_conv = my_conv

        z_pool = F.max_pool2d(z_conv, 2, 2)
        my_pool = pool2d_scalar(z_conv, self.device)
        
        print("TEST!!! pool = ", diff_mse(my_pool, z_pool))
        z_pool = my_pool
       
        z_pool_reshaped = z_pool.view(-1, 20*12*12)

        z_fc1 = self.fc_layer1(z_pool_reshaped)
        my_fc1 = fc_layer_scalar(z_pool_reshaped, self.fc_layer1.weight, self.fc_layer1.bias, self.device)

        print("TEST!!! fc1 = ", diff_mse(z_fc1, my_fc1))
        z_fc1 = my_fc1

        z_relu = F.relu(z_fc1)
        my_relu = relu_scalar(z_fc1, self.device)

        print("TEST!!! relu = ", diff_mse(my_relu, z_relu))
        z_relu = my_relu

        z_fc2 = self.fc_layer2(z_relu)
        my_fc2= fc_layer_scalar(z_relu, self.fc_layer2.weight, self.fc_layer2.bias, self.device)

        print("TEST!!! fc2 = ", diff_mse(my_fc2, z_fc2))
        z_fc2 = my_fc2

        y = F.softmax(z_fc2, dim=1)

        return y


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(torch.log(output), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(args):
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = SimpleConvNet(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    main(args)
