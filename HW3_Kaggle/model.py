import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models.resnet import Bottleneck

nclasses = 20
input_size = 64

pretrained_model_resnet101 = models.resnet101(pretrained=True)


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
