import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ConformalPrediction.create_conformal import create_conformal


Conformal = create_conformal("APS", 0.1, True, True)

probs =  np.random.random((100,10))
probs = probs / np.sum(probs, keepdims=True, axis=1)
targets = np.random.randint(0,10,100)
probs = torch.tensor(probs)
targets = torch.tensor(targets)
dataset_logits = TensorDataset(probs, targets.long()) 

Conformal.solve_qhat(dataset_logits)