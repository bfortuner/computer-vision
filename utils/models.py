import torch
import torch.nn as nn
import torchvision.models


def load_model(fpath, cuda=True):
    if cuda:
        return torch.load(fpath).cuda()
    return torch.load(fpath)


def save_model(model, fpath):
    torch.save(model.cpu(), fpath)


def load_weights(model, fpath):
    state = torch.load(fpath)
    model.load_state_dict(state['state_dict'])


def save_weights(model, fpath, epoch=None, name=None):
    torch.save({
        'name': name,
        'epoch': epoch,
        'state_dict': model.state_dict()
    }, fpath)


def freeze_layers(model, n_layers):
    i = 0
    for child in model.children():
        if i >= n_layers:
            break
        print(i, "freezing", child)
        for param in child.parameters():
            param.requires_grad = False
        i += 1


def freeze_nested_layers(model, n_layers):
    i = 0
    for child in model.children():
        for grandchild in child.children():
            if isinstance(grandchild, torch.nn.modules.container.Sequential):
                for greatgrand in grandchild.children():
                    if i >= n_layers:
                        break
                    for param in greatgrand.parameters():
                        param.requires_grad = False
                    print(i, "freezing", greatgrand)
                    i += 1
            else:
                if i >= n_layers:
                    break
                for param in grandchild.parameters():
                    param.requires_grad = False
                print(i, "freezing", grandchild)
                i += 1


def init_nested_layers(module, conv_init, fc_init):
    for child in module.children():
        if len(list(child.children())) > 0:
            init_nested_layers(child, conv_init, fc_init)
        else:
            init_weights(child, conv_init, fc_init)


def init_weights(layer, conv_init, fc_init):
    if isinstance(layer, torch.nn.Conv2d):
        print("init", layer, "with", conv_init)
        conv_init(layer.weight)
    elif isinstance(layer, torch.nn.Linear):
        print("init", layer, "with", fc_init)
        fc_init(layer.weight)


def cut_model(model, cut):
    return nn.Sequential(*list(model.children())[:cut])

def get_resnet18(pretrained, n_freeze):
    resnet = torchvision.models.resnet18(pretrained)
    if n_freeze > 0:
        freeze_layers(resnet, n_freeze)
    return resnet


def get_resnet34(pretrained, n_freeze):
    resnet = torchvision.models.resnet34(pretrained)
    if n_freeze > 0:
        freeze_layers(resnet, n_freeze)
    return resnet


def get_resnet50(pretrained, n_freeze):
    resnet = torchvision.models.resnet50(pretrained)
    if n_freeze > 0:
        freeze_layers(resnet, n_freeze)
    return resnet

class Resnet(nn.Module):
    def __init__(self, resnet, classifier):
        super().__init__()
        self.resnet = resnet
        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = classifier

    def forward(self, x):
        x = self.resnet(x)
        x = self.ap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_classifier(in_chans, n_classes):
    return nn.Sequential(
        nn.Linear(in_chans, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, n_classes),
        nn.Softmax()
    )
