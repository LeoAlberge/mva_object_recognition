from model import Net
from data import data_transforms
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from torch.optim import lr_scheduler
import pandas as pd
from sklearn.model_selection import ParameterGrid

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()

hyper_parameters = {
    'nb_layer': [1],
    'lr': [0.01, 0.001, 0.0001],
    'sgd_moment': [0.5, 0.9],
    'epochs': [15]
}

# Data initialization and loading

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms['train']),
    batch_size=args.batch_size, shuffle=True, num_workers=1)


val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms['val']),
    batch_size=args.batch_size, shuffle=False, num_workers=1)


# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script


def train(epoch, results, schedule=False, save_results=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(output, target)
        loss.backward()
        if schedule:
            scheduler.step()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            if save_results:
                accuracy_val, loss_val = validation(val_loader, verbose=False)
                accuracy_train, loss_train = validation(train_loader, verbose=False)
                results_temporary = pd.DataFrame([[nb_layer,
                                                   lr,
                                                   sgd_moment,
                                                   epoch,
                                                   batch_idx,
                                                   accuracy_val,
                                                   loss_val,
                                                   accuracy_train,
                                                   loss_train
                                                   ]],
                                                 columns=['nb_layer',
                                                          'sgd_moment',
                                                          'lr',
                                                          'epoch',
                                                          'batch',
                                                          'accuracy_val',
                                                          'loss_val',
                                                          'accuracy_train',
                                                          'loss_train'])
                results = pd.concat([results, results_temporary])
                results.to_csv(args.experiment + '/model_results.csv', index=False)
    return results


def validation(loader, verbose=True):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(loader.dataset)
    accuracy = 100. * float(correct) / len(loader.dataset)
    if verbose:
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
    return accuracy, validation_loss


if __name__ == '__main__':
    results = pd.DataFrame()
    grid = ParameterGrid(hyper_parameters)
    for param in grid:
        nb_layer, lr, sgd_moment = param['nb_layer'], param['lr'], param['sgd_moment']
        model = Net(param['nb_layer'])
        if use_cuda:
            print('Using GPU')
            model.cuda()
        else:
            print('Using CPU')
        optimizer = optim.SGD(model.parameters(), lr=param['lr'], momentum=param['sgd_moment'])
        for epoch in range(1, param['epochs'] + 1):
            results = train(epoch, results, save_results=True)
            validation(val_loader)
