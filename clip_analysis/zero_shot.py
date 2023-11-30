
import os
import time
import os.path as osp
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
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from PIL import Image
from clip import clip
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

######################settings######################
resume = True #True or False
VISUAL_BACKBONE = 'RN50' #'RN50' or 'ViT-B/32'
dataset = 'CIFAR10' #'CIFAR10' or 'Oxford Flowers' or 'MNIST' or 'ImageNet1K'
prompt = 'a photo of a'#a photo of a
batch_size = 128
######################settings######################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##################prepare dataset###################
def prepare_dataset(dataset):
    if(dataset == 'CIFAR10'):
        transform_cifar10_test = transforms.Compose([
    transforms.Resize(size=224),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        test_set = torchvision.datasets.CIFAR10(root='/data/dataset', train=False,download=False, transform=transform_cifar10_test)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        dataset_name = 'CIFAR10'
    

    elif(dataset == 'Oxford Flowers'):
        transform_flowers_test = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4332, 0.3821, 0.2966), (0.2946, 0.2465, 0.2736)),])
        
        
        test_set = torchvision.datasets.Flowers102(root='/data/home/xiezicheng/ML-Hospital', split='test', download=True, transform=transform_flowers_test)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)
        class_names = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 
                       'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', 
                       'colt''s foot', 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 
                       'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower',
                         'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 
                         'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
                           'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy',
                             'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 
                             'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium',
                               'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 
                               'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus',
                                 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple',
                                   'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 
                                   'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 
                       'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 
                       'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily', 'common tulip', 'wild rose']
        dataset_name = 'Oxford Flowers'

    elif(dataset == 'MNIST'):
        transform_mnist_test = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),])
        test_set = torchvision.datasets.MNIST(root='/data/dataset', train=False,download=False, transform=transform_mnist_test)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)
        class_names = ['0','1','2','3','4','5','6','7','8','9']
        #class_names = ['zero','one','two','three','four','five','six','seven','eight','nine']
        dataset_name = 'MNIST'

    elif(dataset == 'ImageNet1K'):
        transform_imagenet_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),])
        test_set = torchvision.datasets.ImageNet(root='/data/dataset', split='val', download=False, transform=transform_imagenet_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)
        class_names = test_set.classes
        dataset_name = 'ImageNet1K'
    
    return test_dataloader, class_names, dataset_name
##################prepare dataset###################

test_dataloader, class_names, dataset_name = prepare_dataset(dataset)
print(len(test_dataloader.dataset))



#torch.save(model, '/data/home/xiezicheng/clip_MIA/save_model/model.pt')

if resume:
    model = torch.load('/data/home/xiezicheng/clip_MIA/save_model/finetuned_model_CIFAR.pt')
    model.to(device)
else:
    model, preprocess = clip.load(name=VISUAL_BACKBONE, device=device, download_root='/data/home/xiezicheng/clip_MIA')
    model.to(device)


def prompt_encode(prompt):
    text_inputs = []
    for i in class_names:
        text_inputs.append(clip.tokenize(prompt + ' ' + i).to(device))
    return torch.cat(text_inputs)

def model_inference(model,image,text_inputs):
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        logits = model(image, text_inputs)
        return logits

def calculate(model, dataloader):
    '''calculate the zero-shot accuracy, precision, recall, f1-score'''
    model.eval()
    with torch.no_grad():
        all_logits = []
        all_labels = []
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model_inference(model, inputs, prompt_encode(prompt))
            all_logits.append(logits[0].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_preds = np.argmax(all_logits, axis=1)
        
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return acc, precision, recall, f1
    
result = calculate(model, test_dataloader)

##################save the result in a txt file###################
with open('/data/home/xiezicheng/clip_MIA/result.txt', 'a+') as f:
    f.write('dataset: ' + dataset_name + '\n')
    f.write('visual backbone: ' + VISUAL_BACKBONE + '\n')
    f.write('prompt: ' + prompt + '\n')
    f.write('zero-shot accuracy: ' + str(result[0]) + '\n')
    f.write('zero-shot precision: ' + str(result[1]) + '\n')
    f.write('zero-shot recall: ' + str(result[2]) + '\n')
    f.write('zero-shot f1-score: ' + str(result[3]) + '\n')
    f.write('\n')
    f.close()
print('finish inference')



