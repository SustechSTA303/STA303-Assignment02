import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from clip import clip
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader


#######################set device######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



######################load model######################
#model_name = ' '
#model = torch.load('/data/home/xiezicheng/clip_MIA/save_model/{model_name}}')
model = torch.load('/data/home/xiezicheng/clip_MIA/save_model/finetuned_model_CIFAR.pt')
model.eval()
model.to(device)
print('model loaded!')


######################settings######################
dataset = 'CIFAR10' #CIFAR10, Oxford Flowers, MNIST, ImageNet1K
prompt = 'a photo of a'#a photo of a
batch_size =64 # 128
few_shot_num = 10000 #注意flowers数据集的训练集很小！
a = 0.5
transform_types = [' ','RandomHorizontalFlip','RandomVerticalFlip','RandomRotation','RandomPerspective','ColorJitter','AddGaussianNoise']
######################settings######################

'''if 'CIFAR10' in model_name:
    dataset = 'CIFAR10'
elif 'Oxford Flowers' in model_name:
    dataset = 'Oxford Flowers'
elif 'MNIST' in model_name:
    dataset = 'MNIST'
elif 'ImageNet1K' in model_name:
    dataset = 'ImageNet1K'
'''



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
        class_names = ['0','1','2','3','4','5','6','7','8','9']
        #class_names = ['zero','one','two','three','four','five','six','seven','eight','nine']
        dataset_name = 'MNIST'
    
    elif(dataset == 'ImageNet1K'):
        transform_imagenet_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),])
        train_set = torchvision.datasets.ImageNet(root='/data/dataset', split='train', download=False, transform=transform_imagenet_test)
        test_set = torchvision.datasets.ImageNet(root='/data/dataset', split='val', download=False, transform=transform_imagenet_test)
        class_names = test_set.classes
        dataset_name = 'ImageNet1K'

    elif(dataset=='CIFAR100'):
        transform_cifar100_test = transforms.Compose([
        transforms.Resize(size=(224)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        test_set = torchvision.datasets.CIFAR100(root='/data/dataset', train=False, download=False, transform=transform_cifar100_test)
        train_set = torchvision.datasets.CIFAR100(root='/data/dataset', train = True, download=False, transform=transform_cifar100_test)
        class_names = test_set.classes
        dataset_name = 'CIFAR100'
   
    member_set = train_set
    #nonmember_set = test_set
    transform_cifar100_test = transforms.Compose([
        transforms.Resize(size=(224)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    nonmember_set = torchvision.datasets.CIFAR100(root='/data/dataset', train=False, download=False, transform=transform_cifar100_test)

    return member_set,nonmember_set, class_names, dataset_name

###############get few-shot data##############
member_set,nonmember_set, class_names, dataset_name = prepare_dataset(dataset)
print('dataset prepared!')


def prompt_encode(prompt,labels):
    text_inputs = []
    for i in labels:
        if isinstance(i, torch.Tensor):
            i = str(i.item())
        else:
            i = str(i)
        text_inputs.append(clip.tokenize(prompt + ' ' + i).to(device))
    return torch.cat(text_inputs)

def calculate_CS(dataloader):
    cs_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            text_inputs = prompt_encode(prompt,targets)
            outputs = model(inputs, text_inputs)
            cs = outputs[0]
            diagonal_elements = torch.diag(cs)
            cs_list.append(diagonal_elements)
    cs_list = torch.cat(cs_list)
    cs_avg = torch.mean(cs_list)
    cs_std = torch.std(cs_list)
    return cs_avg,cs_std

def get_dataloader(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False, num_workers=2)
    return dataloader

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def data_augmentation(dataset,transform_types):
    if transform_types=='RandomHorizontalFlip':
        transform_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        ])
    elif transform_types=='RandomVerticalFlip':
        transform_augmentation = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        ])
    elif transform_types=='RandomRotation':
        transform_augmentation = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        ])
    elif transform_types=='RandomPerspective':
        transform_augmentation = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        transforms.ToTensor(),
        ])
    elif transform_types=='ColorJitter':
        transform_augmentation = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        ])
    elif transform_types=='AddGaussianNoise':
        transform_augmentation = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1),
        ])
    else:
        transform_augmentation = transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset_augmentation = dataset
    dataset_augmentation.transform = transform_augmentation
    return dataset_augmentation

few_shot_dataset,member_test_dataset = torch.utils.data.random_split(member_set, [few_shot_num, len(member_set)-few_shot_num])
few_shot_dataloader = get_dataloader(few_shot_dataset)
print('few shot dataset size:',len(few_shot_dataloader.dataset))

nonmember_train_dataset,nonmember_test_dataset = torch.utils.data.random_split(nonmember_set, [int(len(nonmember_set)*0.5), int(len(nonmember_set)*0.5)])
nonmember_train_dataloader = get_dataloader(nonmember_train_dataset)

few_shot_csavg,few_shot_csstd = calculate_CS(few_shot_dataloader)
print('few_shot_csavg:'+str(few_shot_csavg.item()))
nonmember_csavg,nonmember_csstd = calculate_CS(nonmember_train_dataloader)
print('nonmember_csavg:'+str(nonmember_csavg.item()))

few_shot_cs_sum = 0
nonmember_cs_sum = 0

'''for transform_type in transform_types:
    few_shot_dataset_augmentation = data_augmentation(few_shot_dataset,transform_type)
    few_shot_dataloader_augmentation = get_dataloader(few_shot_dataset_augmentation)
    few_shot_csavg_augmentation,few_shot_csstd_augmentation = calculate_CS(few_shot_dataloader_augmentation)
    few_shot_cs_sum += (few_shot_csavg_augmentation.item()-few_shot_csavg.item()) #为什么绝对值为0

few_shot_cs_sum/=len(few_shot_dataset)
print('few_shot_aug_sum:'+str(few_shot_cs_sum))



for transform_type in transform_types:
    nonmember_dataset_augmentation = data_augmentation(nonmember_train_dataset,transform_type)
    nonmember_dataloader_augmentation = get_dataloader(nonmember_dataset_augmentation)
    nonmember_csavg_augmentation,nonmember_csstd_augmentation = calculate_CS(nonmember_dataloader_augmentation)
    nonmember_cs_sum += (nonmember_csavg_augmentation.item()-nonmember_csavg.item())

nonmember_cs_sum/=len(nonmember_train_dataset)
print('nonmember_aug_sum:'+str(nonmember_cs_sum))'''

#baseline = nonmember_cs_sum+a*(few_shot_cs_sum-nonmember_cs_sum)
baseline = nonmember_csavg.item()+a*(few_shot_csavg.item()-nonmember_csavg.item())
#baseline = few_shot_csavg.item()-3*few_shot_csstd.item()
#baseline = 18.5
print('baseline:'+str(baseline))


####################prepare test set####################
class CustomDataset(Dataset):
    def __init__(self,subset,label):
        self.subset = subset
        self.label = label

    def __getitem__(self, index):
        image, _ = self.subset[index]
        return image, self.label
        
    
    def __len__(self):
        return len(self.subset)

test_data_num = min(len(member_test_dataset),len(nonmember_test_dataset))

member_testset_indices = random.sample(range(len(member_test_dataset)), test_data_num)
nonmember_testset_indices = random.sample(range(len(nonmember_test_dataset)), test_data_num)

member_testset = torch.utils.data.Subset(member_test_dataset, member_testset_indices)
nonmember_testset = torch.utils.data.Subset(nonmember_test_dataset, nonmember_testset_indices)

member_testset_labeled = CustomDataset(member_testset, 1)
nonmember_testset_labeled = CustomDataset(nonmember_testset, 0)
    
test_dataset = torch.utils.data.ConcatDataset([member_testset_labeled, nonmember_testset_labeled])
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
   
def model_inference(model,image,text_inputs):
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        logits = model(image, text_inputs)
        return logits[0]


####################calculate the MIA accuracy, precision, recall, f1-score####################
all_preds = []
all_labels = []

all_cs = []
for batch_idx, (inputs, labels) in enumerate(test_dataloader):
    all_labels.append(labels.cpu().numpy())
    inputs = inputs.to(device)
    labels = labels.to(device)
    logits = model_inference(model, inputs, prompt_encode(prompt,labels))
    for i in range(len(logits)):
        all_cs.append(logits[i][i].item())
        if logits[i][i].item()>baseline:
            all_preds.append(1)
        else:
            all_preds.append(0)

for batch_idx, (inputs, labels) in enumerate(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    logits = model_inference(model, inputs, prompt_encode(prompt,labels))
    for i in range(len(logits)):
        all_cs.append(logits[i][i].item())

#test loader 的标签已经换成了0和1！！！      

all_labels = np.concatenate(all_labels)

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)


print('acc:'+str(acc))
print('precision:'+str(precision))
print('recall:'+str(recall))
print('f1:'+str(f1))

with open('/data/home/xiezicheng/clip_MIA/MIAresult.txt','a') as f:
    f.write('dataset:'+dataset_name+'\n')
    f.write('prompt:'+prompt+'\n')
    f.write('few_shot_num:'+str(few_shot_num)+'\n')
    f.write('a:'+str(a)+'\n')
    f.write('transform_types:'+str(transform_types)+'\n')
    f.write('acc:'+str(acc)+'\n')
    f.write('precision:'+str(precision)+'\n')
    f.write('recall:'+str(recall)+'\n')
    f.write('f1:'+str(f1)+'\n')
    f.write('baseline:'+str(baseline)+'\n')
    f.write('few_shot_cs_sum:'+str(few_shot_cs_sum)+'\n')
    f.write('nonmember_cs_sum:'+str(nonmember_cs_sum)+'\n')
    f.write('\n')
    








