import os
import time
import os.path as osp
import utils
from model_utils import prompt_encode
from model_utils import model_inference
from model_utils import calc_msp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import datasets
from torchvision import transforms
import torchvision
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from PIL import Image
from clip import clip
from tqdm import tqdm

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


#! Hyperparameters
# SEED = 1 
# NUM_CLASS = 10

# Training
BATCH_SIZE = 128
# NUM_EPOCHS = 30
# EVAL_INTERVAL=1
# SAVE_DIR = './log'

# # Optimizer
# LEARNING_RATE = 1e-1
# MOMENTUM = 0.9
# STEP=5
# GAMMA=0.5


#! Load CLIP
VISUAL_BACKBONE = 'ViT-B/16' # RN50, ViT-B/32, ViT-B/16
OOD_METHOD = 'msp' # msp, MaxLogit
class_names = utils.imagenet_class  
model, preprocess = clip.load(name=VISUAL_BACKBONE, device=device, download_root='../data/clip')
model.to(device)      


#! Load test dataloader
# Define the image transform pipeline
transform = transforms.Compose([
    transforms.Resize(size=224),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# test_set 1: iNaturalist
ood_set1 = torchvision.datasets.ImageFolder(root='~/workspace/data/ood_data/iNaturalist', transform=transform)
ood_dataloader1 = torch.utils.data.DataLoader(ood_set1, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# test_set 2: Places
ood_set2 = torchvision.datasets.ImageFolder(root='~/workspace/data/ood_data/Places', transform=transform)
ood_dataloader2 = torch.utils.data.DataLoader(ood_set2, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# test_set 3: SUN
ood_set3 = torchvision.datasets.ImageFolder(root='~/workspace/data/ood_data/SUN', transform=transform)
ood_dataloader3 = torch.utils.data.DataLoader(ood_set3, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# test_set 4: Textures
ood_set4 = torchvision.datasets.ImageFolder(root='~/workspace/data/ood_data/Textures/images', transform=transform)
ood_dataloader4 = torch.utils.data.DataLoader(ood_set4, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# in-distribution test_set: imagenet
in_set = torchvision.datasets.ImageFolder(root='/data/dataset/imagenet/images/val', transform=transform)
in_dataloader = torch.utils.data.DataLoader(in_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

prompt = "a photo of a"

#! Calc the MSP threshold so that the FPR95 is 5% on the in-distribution test set
in_logits = []
for batch in tqdm(in_dataloader):
    image, _ = batch
    image = image.to(device)
    text_inputs = prompt_encode(prompt, class_names)
    logits = model_inference(model, image, text_inputs)
    in_logits.append(logits)
in_logits = torch.cat(in_logits)
in_msp = calc_msp(in_logits)
in_msp = in_msp.cpu().numpy()
threshold = np.percentile(in_msp, 50)
print("MSP 95% threhold:", threshold)


#! Evaluate MSP based OOD detection by the threshold calculated above: Accuracy
ood_logits1 = []
for batch in tqdm(ood_dataloader1):
    image, _ = batch
    image = image.to(device)
    text_inputs = prompt_encode(prompt, class_names)
    logits = model_inference(model, image, text_inputs)
    ood_logits1.append(logits)
ood_logits1 = torch.cat(ood_logits1)
ood_msp1 = calc_msp(ood_logits1)
ood_msp1 = ood_msp1.cpu().numpy()
ood_pred1 = ood_msp1 < threshold
ood_acc1 = np.mean(ood_pred1)
print("Accuracy of iNaturalist:", ood_acc1)







#! 

