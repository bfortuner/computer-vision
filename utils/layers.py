import torch.nn as nn


class CenterCrop(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, img):
        bs, c, h, w = img.size()
        xy1 = (w - self.width) // 2
        xy2 = (h - self.height) // 2
        img = img[:, :, xy2:(xy2 + self.height), xy1:(xy1 + self.width)]
        return img

def conv_relu(in_channels, out_channels, kernel_size=3, stride=1,
              padding=1, bias=True):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias),
        nn.ReLU(inplace=True),
    ]

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]

def linear_bn_relu_drop(in_channels, out_channels, dropout=0.5, bias=False):
    layers = [
        nn.Linear(in_channels, out_channels, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers

def get_fc(in_feat, n_classes, activation=None):
    layers = [
        nn.Linear(in_features=in_feat, out_features=n_classes)
    ]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

def get_classifier(in_feat, n_classes, activation, p=0.5):
    layers = [
        nn.BatchNorm1d(num_features=in_feat),
        nn.Dropout(p),
        nn.Linear(in_features=in_feat, out_features=n_classes),
        activation
    ]
    return nn.Sequential(*layers)

def get_mlp_classifier(in_feat, out_feat, n_classes, activation, p=0.01, p2=0.5):
    layers = [
        nn.BatchNorm1d(num_features=in_feat),
        nn.Dropout(p),
        nn.Linear(in_features=in_feat, out_features=out_feat),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=out_feat),
        nn.Dropout(p2),
        nn.Linear(in_features=out_feat, out_features=n_classes),
        activation
    ]
    return nn.Sequential(*layers)
