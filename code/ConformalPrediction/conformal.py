import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from Utils.Meter import AverageMeter


class ConformalPrediction(nn.Module):

    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, X):
        return self.solve_prediction_sets(X)

    def solve_qhat(self, calib_probs_loader):
        scores = self.solve_nonconformity_score(calib_probs_loader)
        print('Calulating q_hat')
        q_hat = np.quantile(scores, 1-self.alpha, interpolation='higher')
        print(f"Optimal q_hat={q_hat}")
        self.q_hat = q_hat 
    
    def solve_nonconformity_score(self):
        return NotImplementedError
    
    def solve_prediction_sets(self, probability, target):
        return NotImplementedError  