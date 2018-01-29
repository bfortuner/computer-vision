import os
import scipy
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


def predict_batch(net, inputs):
    v = Variable(inputs.cuda(), volatile=True)
    return net(v).data.cpu().numpy()


def get_probabilities(model, loader):
    model.eval()
    return np.vstack(predict_batch(model, data[0]) for data in loader)


def get_predictions(probs, thresholds):
    preds = np.copy(probs)
    preds[preds >= thresholds] = 1
    preds[preds < thresholds] = 0
    return preds.astype('uint8')


def get_argmax(output):
    val,idx = torch.max(output, dim=1)
    return idx.data.cpu().view(-1).numpy()


def get_targets(loader):
    targets = None
    for data in loader:
        if targets is None:
            shape = list(data[1].size())
            shape[0] = 0
            targets = np.empty(shape)
        target = data[1]
        if len(target.size()) == 1:
            target = target.view(-1,1)
        target = target.numpy()
        targets = np.vstack([targets, target])
    return targets


def ensemble_with_method(arr, method):
    if method == c.MEAN:
        return np.mean(arr, axis=0)
    elif method == c.GMEAN:
        return scipy.stats.mstats.gmean(arr, axis=0)
    elif method == c.VOTE:
        return scipy.stats.mode(arr, axis=0)[0][0]
    raise Exception("Operation not found")