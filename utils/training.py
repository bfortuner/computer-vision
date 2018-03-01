import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import math
import time

from . import predictions


class QuickTrainer():
    def __init__(self):
        self.metrics = {
            'loss': {
                'trn':[],
                'tst':[]
            },
            'accuracy': {
                'trn':[],
                'tst':[]
            },
        }

    def run(self, net, trn_loader, tst_loader, criterion, optimizer, epochs):
        for epoch in range(1, epochs+1):
            trn_loss, trn_acc = train(net, trn_loader, criterion, optimizer)
            tst_loss, tst_acc = test(net, tst_loader, criterion)
            print('Epoch %d, TrnLoss: %.3f, TrnAcc: %.3f, TstLoss: %.3f, TstAcc: %.3f' % (
                epoch, trn_loss, trn_acc, tst_loss, tst_acc))
            self.metrics['loss']['trn'].append(trn_loss)
            self.metrics['loss']['tst'].append(tst_loss)
            self.metrics['accuracy']['trn'].append(trn_acc)
            self.metrics['accuracy']['tst'].append(tst_acc)
            # lr = get_learning_rate(optimizer)
            # print(get_metric_msg('trn', self.metrics, 0))
            # print(get_metric_msg('tst', self.metrics, 0))


def train(net, dataloader, criterion, optimizer):
    net.train()
    n_batches = len(dataloader)
    total_loss = 0
    total_acc = 0
    for data in dataloader:
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())

        output = net(inputs)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = predictions.get_argmax(output)
        accuracy = get_accuracy(preds, targets.data.cpu().numpy())

        total_loss += loss.data[0]
        total_acc += accuracy

    mean_loss = total_loss / n_batches
    mean_acc = total_acc / n_batches
    return mean_loss, mean_acc


def get_accuracy(preds, targets):
    correct = np.sum(preds==targets)
    return correct / len(targets)


def test(net, test_loader, criterion):
    net.eval()
    test_loss = 0
    test_acc = 0
    for data in test_loader:
        inputs = Variable(data[0].cuda(), volatile=True)
        target = Variable(data[1].cuda())
        output = net(inputs)
        test_loss += criterion(output, target).data[0]
        pred = predictions.get_argmax(output)
        test_acc += get_accuracy(pred, target.data.cpu().numpy())
    test_loss /= len(test_loader) #n_batches
    test_acc /= len(test_loader)
    return test_loss, test_acc


class Trainer():
    def __init__(self, metrics):
        self.metrics = metrics

    def train(self, model, optim, lr_adjuster, criterion, trn_loader,
              val_loader, n_epochs, n_classes):
        start_epoch = 1
        end_epoch = start_epoch + n_epochs

        for epoch in range(start_epoch, end_epoch):
            current_lr = lr_adjuster.get_learning_rate(optim)

            ### Train ###
            trn_start_time = time.time()
            trn_metrics = train_model(model, trn_loader, optim, criterion,
                                      lr_adjuster, epoch, self.metrics)
            trn_msg = log_trn_msg(trn_start_time, trn_metrics, current_lr, epoch)
            print(trn_msg)

            ### Test ###
            val_start_time = time.time()
            val_metrics = test_model(model, val_loader, criterion, self.metrics, n_classes)
            val_msg = log_val_msg(val_start_time, val_metrics, current_lr)
            print(val_msg)

            ### Adjust Lr ###
            if lr_adjuster.iteration_type == 'epoch':
                lr_adjuster.adjust(optim, epoch+1)


def train_model(model, dataloader, optimizer, criterion,
                lr_adjuster, epoch, metrics):
    model.train()
    n_batches = len(dataloader)
    cur_iter = int((epoch-1) * n_batches)+1
    metric_totals = {m.name:0 for m in metrics}

    for data in dataloader:
        inputs = Variable(data[0].cuda(async=True))
        targets = Variable(data[1].cuda(async=True))

        output = model(inputs)
        model.zero_grad()

        loss = criterion(output, targets)
        loss_data = loss.data[0]
        probs = output.data.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        for metric in metrics:
            score = metric.evaluate(loss_data, preds, probs, None)
            metric_totals[metric.name] += score

        loss.backward()
        optimizer.step()

        if lr_adjuster.iteration_type == 'mini_batch':
            lr_adjuster.adjust(optimizer, cur_iter)
        cur_iter += 1

    for metric in metrics:
        metric_totals[metric.name] /= n_batches

    return metric_totals


def test_model(model, loader, criterion, metrics, n_classes):
    model.eval()

    loss = 0
    probs = np.empty((0, n_classes))
    metric_totals = {m.name:0 for m in metrics}

    for data in loader:
        inputs = Variable(data[0].cuda(async=True), volatile=True)
        targets = Variable(data[1].cuda(async=True), volatile=True)

        output = model(inputs)

        loss += criterion(output, targets).data[0]
        probs = np.vstack([probs, output.data.cpu().numpy()])

    loss /= len(loader)
    preds = np.argmax(probs, axis=1)
    for metric in metrics:
        score = metric.evaluate(loss, preds, probs, None)
        metric_totals[metric.name] = score

    return metric_totals

def early_stop(epoch, best_epoch, patience):
    return (epoch - best_epoch) > patience


def log_trn_msg(start_time, trn_metrics, lr, epoch):
    epoch_msg = 'Epoch {:d}'.format(epoch)
    metric_msg = get_metric_msg('trn', trn_metrics, lr)
    time_msg = get_time_msg(start_time)
    combined = epoch_msg + '\n' + metric_msg + time_msg
    return combined


def log_val_msg(start_time, trn_metrics, lr):
    metric_msg = get_metric_msg('val', trn_metrics, lr)
    time_msg = get_time_msg(start_time)
    combined = metric_msg + time_msg
    return combined


def get_metric_msg(dset, metrics_dict, lr=0):
    msg = dset.capitalize() + ' - '
    for name in metrics_dict.keys():
        print(metrics_dict[name])
        metric_str = ('{:.4f}').format(metrics_dict[name]).lstrip('0')
        msg += ('{:s} {:s} | ').format(name, metric_str)
    msg += 'LR ' + '{:.6f}'.format(lr).rstrip('0').lstrip('0') + ' | '
    return msg


def get_time_msg(start_time):
    time_elapsed = time.time() - start_time
    msg = 'Time {:.1f}m {:.2f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    return msg

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']
