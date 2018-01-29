import operator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
from sklearn import metrics as scipy_metrics
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import warnings



class Metric():
    def __init__(self, name, minimize=True):
        self.name = name
        self.minimize = minimize

    def get_best_epoch(self, values):
        if self.minimize:
            idx, value = min(enumerate(values),
                key=operator.itemgetter(1))
        else:
            idx, value = max(enumerate(values),
                key=operator.itemgetter(1))
        epoch = idx + 1 # epochs start at 1
        return epoch, value

    def evaluate(self, loss, preds, probs, targets):
        pass

    def format(self, value):
        pass


class AuxiliaryMetric():
    def __init__(self, name, units):
        self.name = name
        self.units = units


class Accuracy(Metric):
    def __init__(self):
        super().__init__('Accuracy', minimize=False)

    def evaluate(self, loss, preds, probs, targets):
        return get_accuracy(preds, targets)

    def format(self, value):
        return value


class Loss(Metric):
    def __init__(self):
        super().__init__('Loss', minimize=True)

    def evaluate(self, loss, preds, probs, targets):
        return loss

    def format(self, value):
        return value


class F2Score(Metric):
    def __init__(self, target_threshold=None):
        super().__init__('F2', minimize=False)
        self.target_threshold = target_threshold  # pseudo soft targets

    def evaluate(self, loss, preds, probs, targets):
        average = 'samples' if targets.shape[1] > 1 else 'binary'
        if self.target_threshold is not None:
            targets = targets > self.target_threshold

        return get_f2_score(preds, targets, average)

    def format(self, value):
        return value


class DiceScore(Metric):
    def __init__(self):
        super().__init__('Dice', minimize=False)

    def evaluate(self, loss, preds, probs, targets):
        return get_dice_score(preds, targets)

    def format(self, value):
        return value


def get_accuracy(preds, targets):
    preds = preds.flatten() 
    targets = targets.flatten()
    correct = np.sum(preds==targets)
    return correct / len(targets)


def get_cross_entropy_loss(probs, targets):
    return F.binary_cross_entropy(
              Variable(torch.from_numpy(probs)),
              Variable(torch.from_numpy(targets).float())).data[0]


def get_recall(preds, targets):
    return scipy_metrics.recall_score(targets.flatten(), preds.flatten())


def get_precision(preds, targets):
    return scipy_metrics.precision_score(targets.flatten(), preds.flatten())


def get_roc_score(probs, targets):
    return scipy_metrics.roc_auc_score(targets.flatten(), probs.flatten())


def get_dice_score(preds, targets):
    eps = 1e-7
    batch_size = preds.shape[0]
    preds = preds.reshape(batch_size, -1)
    targets = targets.reshape(batch_size, -1)

    total = preds.sum(1) + targets.sum(1) + eps
    intersection = (preds * targets).astype(float)
    score = 2. * intersection.sum(1) / total
    return np.mean(score)


def get_f2_score(y_pred, y_true, average='samples'):
    y_pred, y_true, = np.array(y_pred), np.array(y_true)
    return fbeta_score(y_true, y_pred, beta=2, average=average) 
