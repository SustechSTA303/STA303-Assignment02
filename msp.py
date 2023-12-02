import const
from model_utils import calc_msp_threshold
from model_utils import eval_msp
from torchvision import transforms
import torchvision
import torch
from clip import clip
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--vb', type=str, default='RN50', help='RN50, ViT-B/32, ViT-B/16')
parser.add_argument('--percentage', type=int, default=5, help='the percentage of the in-distribution test set be misclassified as OOD')
parser.add_argument('--cuda', type=int, default=5, help='cuda device')
args = parser.parse_args()
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

#! Hyperparameters
# Training
BATCH_SIZE = 128


#! Load CLIP
VISUAL_BACKBONE = args.vb # RN50, ViT-B/32, ViT-B/16
OOD_METHOD = 'msp' # msp, MaxLogit
class_names = const.imagenet_class  
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
# test_set 4: Textures
ood_set2 = torchvision.datasets.ImageFolder(root='~/workspace/data/ood_data/Textures/images', transform=transform)
ood_dataloader2 = torch.utils.data.DataLoader(ood_set2, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# test_set 2: Places
ood_set3 = torchvision.datasets.ImageFolder(root='~/workspace/data/ood_data/Places', transform=transform)
ood_dataloader3 = torch.utils.data.DataLoader(ood_set3, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# test_set 3: SUN
ood_set4 = torchvision.datasets.ImageFolder(root='~/workspace/data/ood_data/SUN', transform=transform)
ood_dataloader4 = torch.utils.data.DataLoader(ood_set4, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# in-distribution test_set: imagenet
in_set = torchvision.datasets.ImageFolder(root='/data/dataset/imagenet/images/val', transform=transform)
in_dataloader = torch.utils.data.DataLoader(in_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


#! Prompt and percentage
prompt = "a photo of a"
percentage = args.percentage


#! Calc the MSP threshold so that the FPR95 is 5% on the in-distribution test set
threshold, in_acc = calc_msp_threshold(in_dataloader, model, prompt, class_names, percentage, device)


#! Evaluate MSP based OOD detection by the threshold calculated above: Accuracy
ood_acc1 = eval_msp(ood_dataloader1, model, prompt, class_names, threshold, device, 'iNaturalist')
ood_acc2 = eval_msp(ood_dataloader2, model, prompt, class_names, threshold, device, 'Textures')
ood_acc3 = eval_msp(ood_dataloader3, model, prompt, class_names, threshold, device, 'Places')
ood_acc4 = eval_msp(ood_dataloader4, model, prompt, class_names, threshold, device, 'SUN')


#! Save the results
df = pd.DataFrame({'threshold': [threshold], 'in_acc': [in_acc], 'iNaturalist': [ood_acc1], 'Textures': [ood_acc2], 'Places': [ood_acc3], 'SUN': [ood_acc4]}, index=[0])
if VISUAL_BACKBONE == 'RN50':
    df.to_csv(f'./log/msp/RN50_{percentage}.csv', index=False)
elif VISUAL_BACKBONE == 'ViT-B/32':
    df.to_csv(f'./log/msp/ViT-B_32_{percentage}.csv', index=False)
elif VISUAL_BACKBONE == 'ViT-B/16':
    df.to_csv(f'./log/msp/ViT-B_16_{percentage}.csv', index=False)