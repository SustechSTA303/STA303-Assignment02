import os
import csv
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from Utils.Meter import AverageMeter


def get_logits_targets(model, loader, num_classes):
    logits = torch.zeros((len(loader.dataset), num_classes)) 
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Computing logits for model.')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.cuda()).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    dataset_logits = TensorDataset(logits, labels.long()) 
    return dataset_logits


def get_prob_targets(model, loader, num_classes, T=1.0):
    probs_data = torch.zeros((len(loader.dataset), num_classes)) 
    labels_data = torch.zeros((len(loader.dataset),))
    i = 0
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for probs, targets in tqdm(loader):
            batch_logits = model(probs.cuda()).detach().cpu().to(torch.float32)
            batch_prob = softmax(batch_logits / T)

            probs_data[i:(i+batch_prob.shape[0]), :] = batch_prob
            labels_data[i:(i+batch_prob.shape[0])] = targets.cpu()
            i = i + batch_prob.shape[0]
    # Construct the dataset
    dataset_probs = TensorDataset(probs_data, labels_data.long()) 
    probs_loader = DataLoader(dataset_probs, batch_size=128, shuffle=False)
    return probs_loader


def sort_sum(scores):
    I = scores.argsort(axis=1)[:,::-1]
    ordered = np.sort(scores,axis=1)[:,::-1]
    cumsum = np.cumsum(ordered,axis=1) 
    return I, ordered, cumsum


def solve_metric(conformal, val_probs_loader, print_bool=True):
    print('Calulating Metric')
    with torch.no_grad():
        top1_meter = AverageMeter('top1')
        top5_meter = AverageMeter('top5')
        coverage = AverageMeter('coverage')
        size = AverageMeter('size')
        for probs, targets in tqdm(val_probs_loader):
            targets = targets.cuda()
            probs = probs.cuda()
            pred_set = conformal(probs)

            top1, top5 = accuracy(probs, targets, top_k=(1,5))
            cvg, sz, size_list = coverage_size(pred_set,targets)
            size_list = np.array(size_list)

            top1_meter.update(top1.item()/100.0, n=probs.shape[0])
            top5_meter.update(top5.item()/100.0, n=probs.shape[0])
            coverage.update(cvg, n=probs.shape[0])
            size.update(sz, n=probs.shape[0])

        if print_bool == True:  
            print(f'top1: {top1_meter.avg:.3f} | top5: {top5_meter.avg} | coverage: {coverage.avg:.3f} | size: {size.avg:.3f}')

    return top1_meter.avg, top5_meter.avg, coverage.avg, size.avg


def accuracy(logits, targets, top_k=(1,)):
    k_maprobs = max(top_k)
    batch_size = targets.size(0)

    # 获取top_1到top_k的次序
    _, order = logits.topk(k_maprobs, dim=1, largest=True, sorted=True)
    order = order.t()
    correct = order.eq(targets.view(1,-1).expand_as(order))

    acc = []
    for k in top_k:
        correct_k = correct[:k].float().sum()
        acc.append(correct_k.mul_(100.0 / batch_size))
    return acc


def coverage_size(pred_set, targets):
    covered = 0
    size = 0
    size_list = []
    for i in range(targets.shape[0]):
        cur_target = targets[i].item()
        cur_pred_set = pred_set[i][0]
        if (cur_target in cur_pred_set):
            covered += 1
        size = size + len(cur_pred_set)
        size_list.append(len(cur_pred_set))
    return float(covered)/targets.shape[0], size/targets.shape[0], size_list


def write(vision_backbone, top1, top5, coverage, size, num_calib, file_path=None):
    if vision_backbone == "ViT-B/16":
        vision_backbone = "ViT-B-16"
    elif vision_backbone == "ViT-B/32":
        vision_backbone = "ViT-B-32"

    if file_path == None:
        file_path = f"experiments/results/{vision_backbone}.csv"
    
    if os.path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([top1, top5, coverage, size, num_calib])  
    else:
        columns = ['top1', 'top5', 'coverage', 'size', 'num_calib']
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerow([top1, top5, coverage, size, num_calib])