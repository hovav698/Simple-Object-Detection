import torch
from torch import nn

import torchvision.models as models

import config


class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()
        vgg = models.vgg19(pretrained=True)
        modules = list(vgg.children())[:-1]
        self.vgg = torch.nn.Sequential(*modules)

        # change the VGG19 head output
        self.vgg.add_module('flatten', nn.Flatten())

        # the object localization output
        self.lin1 = nn.Linear(25088, 4)

        # the object classification output
        self.lin2 = nn.Linear(25088, config.num_classes)

        # the object appear flag output
        self.lin3 = nn.Linear(25088, 1)

    def forward(self, img):
        vgg_out = self.vgg(img)

        x1 = nn.Sigmoid()(self.lin1(vgg_out))
        x2 = self.lin2(vgg_out)
        x3 = nn.Sigmoid()(self.lin3(vgg_out))

        output = torch.cat([x1, x2, x3], 1)

        return output
