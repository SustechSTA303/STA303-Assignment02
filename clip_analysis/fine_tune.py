
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
from clip import clip
from clip import loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
#sys.path.append('/data/home/xiezicheng/FLYP')
#from FLYP.src.models.modeling import CLIPEncoder
#from FLYP.src.args import parse_arguments
import matplotlib.pyplot as plt
from clip.loss import ClipLoss


######################settings######################
resume = False #True or False
VISUAL_BACKBONE = 'ViT-B/32' #'RN50' or 'ViT-B/32'
dataset = 'Oxford Flowers' #'CIFAR10' or 'Oxford Flowers' or 'MNIST' or 'ImageNet1K'
prompt = 'a photo of a'#a photo of a
batch_size =64 # 128
lr = 0.001 #0.001
epoch = 30
save = True
logit_scale = 0.07
#args = parse_arguments()
######################settings######################

model_name = f"model_{VISUAL_BACKBONE}_{dataset}_prompt_{prompt.replace(' ', '_')}_bs{batch_size}_lr{lr}_epoch{epoch}_scale{logit_scale}"

device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.cuda.set_device(0)
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


##################prepare dataset###################
def prepare_dataset(dataset):
    if(dataset == 'CIFAR10'):
        transform_cifar10_test = transforms.Compose([
    transforms.Resize(size=(224)),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        test_set = torchvision.datasets.CIFAR10(root='/data/dataset', train=False, download=False, transform=transform_cifar10_test)
        train_set = torchvision.datasets.CIFAR10(root='/data/dataset', train = True, download=False, transform=transform_cifar10_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=2,drop_last=True)
        print('train dataset size: ' + str(len(train_loader.dataset)))
        print('test dataset size: ' + str(len(test_loader.dataset)))
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        dataset_name = 'CIFAR10'

    elif(dataset == 'Oxford Flowers'):
        transform_flowers_test = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4332, 0.3821, 0.2966), (0.2946, 0.2465, 0.2736)),])
        
        train_set = torchvision.datasets.Flowers102(root='/data/home/xiezicheng/ML-Hospital', split='train', download=False, transform=transform_flowers_test)
        test_set = torchvision.datasets.Flowers102(root='/data/home/xiezicheng/ML-Hospital', split='val', download=False, transform=transform_flowers_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=2,drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)

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
        train_set = torchvision.datasets.MNIST(root='/data/dataset', train=True,download=False, transform=transform_mnist_test)
        test_set = torchvision.datasets.MNIST(root='/data/dataset', train=False,download=False, transform=transform_mnist_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=2,drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)
        class_names = ['0','1','2','3','4','5','6','7','8','9']
        #class_names = ['zero','one','two','three','four','five','six','seven','eight','nine']
        dataset_name = 'MNIST'
    
    elif(dataset == 'ImageNet1K'):
        transform_imagenet_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),])
        train_set = torchvision.datasets.ImageNet(root='/data/dataset', split='train', download=False, transform=transform_imagenet_test)
        test_set = torchvision.datasets.ImageNet(root='/data/dataset', split='val', download=False, transform=transform_imagenet_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=2,drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)
        class_names = test_set.classes
        dataset_name = 'ImageNet1K'
    
    elif(dataset =='CIFAR100'):
        transform_cifar100_test = transforms.Compose([
        transforms.Resize(size=(224)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)),])
        test_set = torchvision.datasets.CIFAR100(root='/data/dataset', train=False, download=False, transform=transform_cifar100_test)
        train_set = torchvision.datasets.CIFAR100(root='/data/dataset', train = True, download=False, transform=transform_cifar100_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=2)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=2,drop_last=True)
        class_names = test_set.classes
        dataset_name = 'CIFAR100'

    return train_loader,test_loader, class_names, dataset_name
##################prepare dataset###################

train_loader,test_loader, class_names, dataset_name = prepare_dataset(dataset)
print('dataset prepared!')



with open('/data/home/xiezicheng/clip_MIA/log.txt', 'a+') as f:
    f.write('dataset: ' + dataset_name + '\n')
    f.write('visual backbone: ' + VISUAL_BACKBONE + '\n')
    f.write('prompt: ' + prompt + '\n')
    f.write('learning rate: ' + str(lr) + '\n')
    f.write('total epoch: ' + str(epoch) + '\n')
    f.write('\n')
    f.close()

if resume:
    model = torch.load('/data/home/xiezicheng/clip_MIA/save_model/model.pt')
    model.to(device)
else:
    model, preprocess = clip.load(name=VISUAL_BACKBONE, device=device, download_root='/data/home/xiezicheng/clip_MIA')

def prompt_encode(prompt,labels):
    text_inputs = []
    for i in labels:
        if isinstance(i, torch.Tensor):
            i = str(i.item())
        else:
            i = str(i)
        text_inputs.append(clip.tokenize(prompt + ' ' + i).to(device))
    return torch.cat(text_inputs)
    

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    image_features = model.encode_image(images)
    text_features = model.encode_text(prompt_encode(prompt,labels))
    print('image_features_shape:'+str(image_features.shape))
    print('text_features_shape:'+str(text_features.shape))
    image_features,text_features = model(images, prompt_encode(prompt,labels))
    print('image_features_shape:'+str(image_features.shape))
    print('text_features_shape:'+str(text_features.shape))
    break
    
    #model = CLIPEncoder(args=args,keep_lang=True)

model.to(device)
print('model prepared!')

#optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
optimizer = optim.SGD(model.parameters(), lr=lr)

##################loss###################
'''contrastive_loss = ClipLoss(local_loss=False,
                            gather_with_grad=False,
                            cache_labels=True,
                            rank=0,
                            world_size=1,
                            use_horovod=False)'''

def contrastive_loss(image_features,text_features,ground_labels,logit_scale=0.07):
    
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    #logits = (100.0 * image_features @ text_features.T) / temperature
    logits_per_image = np.exp(logit_scale) * image_features @ text_features.T
    logits_per_text = np.exp(logit_scale) * text_features @ image_features.T

    #print(logits_per_image)
    #print(logits_per_text)

    ground_labels_repeated = ground_labels.view(1, -1).repeat(image_features.shape[0], 1)
    equal_labels = (ground_labels_repeated == ground_labels.view(-1, 1)).type(torch.float)

    labels = equal_labels / torch.sum(equal_labels,dim=1).view(-1,1)
    total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

    return total_loss

t_start = time.time()

def calculate(model, dataloader):
    model.eval()
    with torch.no_grad():
        all_logits = []
        all_labels = []
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs, prompt_encode(prompt,class_names))
            all_logits.append(logits[0].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_preds = np.argmax(all_logits, axis=1)
        
        acc = accuracy_score(all_labels, all_preds)
        return acc

def evaluate_model(model, data_loader, loss_fn):
    model.eval()  
    total_loss = 0

    with torch.no_grad():  
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            image_features = model.encode_image(images)
            text_features = model.encode_text(prompt_encode(prompt,labels))
    
            loss = loss_fn(image_features, text_features)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

##################schedular###################
def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr=0.0):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr + min_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster
##################start finetuning###################
try:

    num_batches = len(train_loader)
    scheduler = cosine_lr(optimizer, lr, 500,
                          epoch * num_batches, 0)
    print('zero shot acc:'+str(calculate(model, test_loader)))
    print('start finetuning!')
    finetune_iter = iter(train_loader)

    
    
    for e in range(epoch):
        loss_sum = 0
        model.train()


        for images,labels in train_loader:
            
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            #image_features,text_features,logit_scale2 = model(images, prompt_encode(prompt,labels))
            #image_features,text_features = model(images, prompt_encode(prompt,labels))
            image_features = model.encode_image(images)
            text_features = model.encode_text(prompt_encode(prompt,labels))
            #print(image_features)
            #print(text_features)

            loss = contrastive_loss(image_features=image_features,text_features=text_features,logit_scale=logit_scale,ground_labels=labels)
            #print(loss.item())
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        
        loss_avg = loss_sum/num_batches
        test_acc = calculate(model, test_loader)
        train_acc = calculate(model, train_loader)
        train_loss = loss_avg

        output_str = 'epoch: {} train acc: {} test acc: {} train loss: {} total time: {} total sample: {}\n'.format(e+1, train_acc,test_acc, train_loss,time.time() - t_start,str(len(train_loader.dataset)))
        print(output_str)
        with open('/data/home/xiezicheng/clip_MIA/log.txt', 'a+') as f:
            f.write('epoch: ' + str(e+1) + '\n')
            f.write('train acc: ' + str(train_acc) + '\n')
            f.write('test acc: ' + str(test_acc) + '\n')
            f.write('train loss: ' + str(train_loss) + '\n')
            f.write('\n')
            f.close()


    """for e in range(epoch):
        model.train()
        for batch in tqdm(train_loader):


            model.zero_grad()
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(prompt_encode(prompt,labels))

            loss = contrastive_loss(image_features=image_features,text_features=text_features)
            loss.backward()
            optimizer.step()

        test_acc = calculate(model, test_loader)
        train_acc = calculate(model, train_loader)
        test_loss = evaluate_model(model, test_loader,contrastive_loss)
        train_loss = evaluate_model(model, train_loader,contrastive_loss)

        output_str = 'epoch: {} train acc: {} test acc: {} train loss: {} test loss: {} total time: {} total sample: {}\n'.format(e+1, train_acc,test_acc, train_loss,test_loss,time.time() - t_start,str(len(train_loader.dataset)))
        print(output_str)
        with open('/data/home/xiezicheng/clip_MIA/log.txt', 'a+') as f:
            f.write('epoch: ' + str(e) + '\n')
            f.write('train acc: ' + str(train_acc) + '\n')
            f.write('test acc: ' + str(test_acc) + '\n')
            f.write('train loss: ' + str(train_loss) + '\n')
            f.write('test loss: ' + str(test_loss) + '\n')
            f.write('\n')
            f.close()"""
    
except KeyboardInterrupt:
    pass

print('finish finetuning!')

if(save):
    torch.save(model, '/data/home/xiezicheng/clip_MIA/save_model/finetuned_model.pt')
