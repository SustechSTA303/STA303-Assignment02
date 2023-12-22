import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch 
import torchvision.transforms as transforms

from Data.create_dataset import get_data
from Model.model import CLIP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
prompt = "This is a"
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model = CLIP('ViT-B/16', prompt, class_names)

transform_cifar10_test = transforms.Compose([
    transforms.Resize(size=224),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
num_calib, train_dataset, val_dataset = get_data("Cifar10", transform_cifar10_test)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, pin_memory=True)

with torch.no_grad():
    model.eval()

    val_loss = 0.0
    val_corrects = 0

    for batch_idx, (image, target) in enumerate(val_loader):

        image = image.to(device)
        target = target.to(device)

        # test model
        logits = model(image)
        _, preds = torch.max(logits, 1)
        
        val_corrects += torch.sum(preds == target.data)
        print(logits)
        break