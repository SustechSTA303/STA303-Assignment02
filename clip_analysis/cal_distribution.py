import torch
import torchvision
import torchvision.transforms as transforms
from clip import clip
import numpy as np
import random
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
######################load model######################
#model_name = ' '
#model = torch.load('/data/home/xiezicheng/clip_MIA/save_model/{model_name}}')
#model = torch.load('/data/home/xiezicheng/clip_MIA/save_model/finetuned_model_CIFAR.pt')
model, preprocess = clip.load(name='ViT-B/32', device=device, download_root='/data/home/xiezicheng/clip_MIA')
model.eval()
print('model loaded!')


######################settings######################
dataset = 'CIFAR10' #CIFAR10, Oxford Flowers, MNIST, ImageNet1K
prompt = 'a photo of a'#a photo of a
batch_size =64 # 128
few_shot_num = 20 #注意flowers数据集的训练集很小！
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

    elif(dataset =='CIFAR100'):
        transform_cifar100_test = transforms.Compose([
        transforms.Resize(size=(224)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),])
        train_set = torchvision.datasets.CIFAR100(root='/data/dataset', train=True,download=False, transform=transform_cifar100_test)
        test_set = torchvision.datasets.CIFAR100(root='/data/dataset', train=False,download=False, transform=transform_cifar100_test)
        class_names = test_set.classes
        dataset_name = 'CIFAR100'
        
    member_set = train_set
    nonmember_set = test_set
    transform_mnist_test = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),])
    #nonmember_set = torchvision.datasets.MNIST(root='/data/dataset', train=False,download=False, transform=transform_mnist_test)

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

'''few_shot_csavg,few_shot_csstd = calculate_CS(few_shot_dataloader)
print('few_shot_csavg:'+str(few_shot_csavg.item()))
nonmember_csavg,nonmember_csstd = calculate_CS(nonmember_train_dataloader)
print('nonmember_csavg:'+str(nonmember_csavg.item()))

few_shot_cs_sum = 0

for transform_type in transform_types:
    few_shot_dataset_augmentation = data_augmentation(few_shot_dataset,transform_type)
    few_shot_dataloader_augmentation = get_dataloader(few_shot_dataset_augmentation)
    few_shot_csavg_augmentation,few_shot_csstd_augmentation = calculate_CS(few_shot_dataloader_augmentation)
    few_shot_cs_sum += few_shot_csavg_augmentation.item()-few_shot_csavg.item() #为什么绝对值为0

few_shot_cs_sum/=len(few_shot_dataset)
print('few_shot_aug_sum:'+str(few_shot_cs_sum))

nonmember_cs_sum = 0

for transform_type in transform_types:
    nonmember_dataset_augmentation = data_augmentation(nonmember_train_dataset,transform_type)
    nonmember_dataloader_augmentation = get_dataloader(nonmember_dataset_augmentation)
    nonmember_csavg_augmentation,nonmember_csstd_augmentation = calculate_CS(nonmember_dataloader_augmentation)
    nonmember_cs_sum += nonmember_csavg_augmentation.item()-nonmember_csavg.item()

nonmember_cs_sum/=len(nonmember_train_dataset)
print('nonmember_aug_sum:'+str(nonmember_cs_sum))'''


####################prepare test set####################


test_data_num = min(len(member_test_dataset),len(nonmember_test_dataset))

member_testset_indices = random.sample(range(len(member_test_dataset)), test_data_num)
nonmember_testset_indices = random.sample(range(len(nonmember_test_dataset)), test_data_num)

member_testset = torch.utils.data.Subset(member_test_dataset, member_testset_indices)
nonmember_testset = torch.utils.data.Subset(nonmember_test_dataset, nonmember_testset_indices)

member_testset_dataloader = get_dataloader(member_testset)
nonmember_testset_dataloader = get_dataloader(nonmember_testset)
   
def model_inference(model,image,text_inputs):
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        logits = model(image, text_inputs)
        return logits[0]


####################calculate distribution####################
def get_all_cs(dataloader):
    all_cs = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model_inference(model, inputs, prompt_encode(prompt,labels))
        for i in range(len(logits)):
            all_cs.append(logits[i][i].item())
    return all_cs


def plot(dataloader,plot_name):
    all_cs = get_all_cs(dataloader)
    unique_elements, counts = np.unique(all_cs, return_counts=True)
    prob = counts/counts.sum()

    random_sample = np.random.choice(unique_elements, size=int(len(dataloader.dataset)/2), p=prob)
    print('sample size:',len(random_sample))

    bins = np.arange(15,30,0.005)

    print('hist!')
    plt.hist(random_sample, bins=bins, density=True, alpha=0.5)
    plt.xlabel('CS')
    plt.ylabel('Probability')
    plt.title(dataset+' '+plot_name)

    plt.savefig('/data/home/xiezicheng/clip_MIA/save_figure/'+plot_name+'.png')
    #plt.close()

plot(member_testset_dataloader,'member_testset')
plot(nonmember_testset_dataloader,'nonmember_testset')
plot(few_shot_dataloader,'few_shot_dataset')

    








