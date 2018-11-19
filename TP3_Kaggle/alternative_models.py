import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models.resnet import BasicBlock, Bottleneck

nclasses = 20
input_size = 64

pretrained_model_resnet18 = models.resnet18(pretrained=True)
pretrained_model_resnet101 = models.resnet101(pretrained=True)
pretrained_model_alexnet = models.alexnet(pretrained=True)


class NetAlex(models.AlexNet):
    def __init__(self):
        super(Net, self).__init__(num_classes=nclasses)
        self.get_weitghts_from_pretrained(pretrained_model_alexnet)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 320),
            nn.ReLU(inplace=True),
            nn.Linear(320, 20),
        )

    def get_weitghts_from_pretrained(self, pretrained_model):
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items(
        ) if k in model_dict and not(k.startswith('classifier.6'))}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict, strict=False)


class NetOld(models.ResNet):
    """
    Model inherating from ResNet 18.
    The model used pretrained weights.
    Severals layers can be frozen.
    """

    def __init__(self, nb_layers):
        super(Net, self).__init__(BasicBlock, [2, 2, 2, 2])
        # getting weights from pretrained model
        self.get_weitghts_from_pretrained(pretrained_model_resnet18)
        # freezing layers
        self.freeze_layers(nb_layers)

    def get_weitghts_from_pretrained(self, pretrained_model):
        """
        Update Net with pretrained weights.
        """
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)

    def freeze_layers(self, nb_layers):
        if nb_layers == 1:
            # freeze all layers
            for name, param in self.named_parameters():
                param.requires_grad = False
            # reinitalizing fully linear layer
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, nclasses)

        if nb_layers == 2:
            # freeze layers except layer 4
            for name, param in self.named_parameters():
                if not(name.startswith('layer4')):
                    param.requires_grad = False
            # reinitalizing fully linear layer
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, nclasses)

        if nb_layers == 3:
            # freeze layers except 3 and 4
            for name, param in self.named_parameters():
                if not(name.startswith('layer4')) and not(name.startswith('layer3')):
                    param.requires_grad = False


class Net(models.ResNet):
    """
    Model inherating from ResNet 101.
    The model used pretrained weights.
    Severals layers can be frozen.
    """

    def __init__(self, nb_layers):
        super(Net, self).__init__(Bottleneck, [3, 4, 23, 3])
        # getting weights from pretrained model
        self.get_weitghts_from_pretrained(pretrained_model_resnet101)
        # freezing layers
        self.freeze_layers(nb_layers)

    def get_weitghts_from_pretrained(self, pretrained_model):
        """
        Update Net with pretrained weights.
        """
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)

    def freeze_layers(self, nb_layers):
        if nb_layers == 1:
            # freeze all layers
            for name, param in self.named_parameters():
                param.requires_grad = False
            # reinitalizing fully linear layer
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, nclasses)

        if nb_layers == 2:
            # freeze layers except layer 4
            for name, param in self.named_parameters():
                if not(name.startswith('layer4')):
                    param.requires_grad = False
            # reinitalizing fully linear layer
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, nclasses)

        if nb_layers == 3:
            # freeze layers except 3 and 4
            for name, param in self.named_parameters():
                if not(name.startswith('layer4')) and not(name.startswith('layer3')):
                    param.requires_grad = False


class NetGeneral(nn.Module):
    """
    The model used pretrained weights.
    Severals layers can be frozen.
    """

    def __init__(self, layers_to_remove):
        super(NetGeneral, self).__init__()
        net = pretrained_model_resnet101
        if layers_to_remove == 0:
            module = nn.Sequential(*list(pretrained_model_resnet101.children())[:-2])
            self.add_module('feature_module', module)
            self.add_module('avg_pool', nn.AvgPool2d(kernel_size=7, stride=1, padding=0))
            self.add_module('fc', nn.Linear(2048, nclasses))

        elif layers_to_remove == 1:
            module = nn.Sequential(*list(pretrained_model_resnet101.children())[:-3])
            self.add_module('feature_module', module)
            self.add_module('avg_pool', nn.AvgPool2d(kernel_size=14, stride=1, padding=0))
            self.add_module('fc', nn.Linear(1024, nclasses))

        elif layers_to_remove == 2:
            module = nn.Sequential(*list(pretrained_model_resnet101.children())[:-4])
            self.add_module('feature_module', module)
            self.add_module('avg_pool', nn.AvgPool2d(kernel_size=28, stride=1, padding=0))
            self.add_module('fc', nn.Linear(512, nclasses))

        self.freeze_layers()

    def freeze_layers(self, all_except_linear=True):
        if (all_except_linear):
            for name, param in self.named_parameters():
                if not(name.startswith('fc')):
                    param.requires_grad = False

    def forward(self, x):
        x = self.feature_module(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
