import zipfile
import os
import PIL
import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

r_mean = 0.496
g_mean = 0.517
b_mean = 0.436

r_std = 0.232
g_std = 0.229
b_std = 0.264

data_transfom_identity = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15, translate=None, scale=[
                                0.5, 2], shear=PIL.Image.NEAREST, resample=False, fillcolor=0),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[r_mean, g_mean, b_mean],
                             std=[r_std, g_std, b_std])]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[r_mean, g_mean, b_mean],
                             std=[r_std, g_std, b_std])
    ])
}
