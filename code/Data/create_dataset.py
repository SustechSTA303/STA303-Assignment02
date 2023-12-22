import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_data(dataset, transform=None, root=None):

    if dataset == 'ImageNet':
        return get_ImageNet_data(root)
    
    elif dataset == 'Cifar10':
        return get_Cifar10_data(transform, root)
    
    elif dataset == 'Cifar100':
        return get_Cifar100_data(transform, root)
    
    else:
        return NotImplementedError
    

def get_ImageNet_data(root=None):

    if root == None:
            traindir = '/data/dataset/imagenet/images/train/'
            valdir = '/data/dataset/imagenet/images/val/'
    else :
        traindir = f'{root}train/'
        valdir = f'{root}val/'   

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    return 10000, train_dataset, val_dataset


def get_Cifar10_data(transform=None, root=None):

    if root == None:
        root = r'/data/dataset/'

    if transform is None:
        normalize = transforms.Normalize(mean=[0.492, 0.482, 0.446],
                                        std=[0.247, 0.244, 0.262])

        train_dataset = datasets.CIFAR10(root=root, 
                                         train=True, 
                                         download=True, 
                                         transform=transforms.Compose([transforms.RandomHorizontalFlip(), 
                                                                        transforms.RandomCrop(32, padding=4),
                                                                        transforms.ToTensor(), 
                                                                        normalize]))
        val_dataset = datasets.CIFAR10(root=root, 
                                       train=False, 
                                       download=True, 
                                       transform=transforms.Compose([transforms.ToTensor(), normalize]))
    else:
        train_dataset = datasets.CIFAR10(root=root, 
                                        train=True, 
                                        download=True, 
                                        transform=transform)
        val_dataset = datasets.CIFAR10(root=root, 
                                       train=False, 
                                       download=True, 
                                       transform=transform)

    return 5000, train_dataset, val_dataset


def get_Cifar100_data(transform=None, root=None):

    if root == None:
        root = r'/data/dataset/'

    if transform is None:
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        normalize = transforms.Normalize(mean=mean, std=std)

        train_dataset = datasets.CIFAR100(root=root,
                                        train=True,
                                        download=False,
                                        transform=None)
        val_dataset = datasets.CIFAR100(root=root,
                                        train=False,
                                        download=False,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    else:
        train_dataset = datasets.CIFAR100(root=root,
                                        train=True,
                                        download=False,
                                        transform=transform)
        val_dataset = datasets.CIFAR100(root=root,
                                        train=False,
                                        download=False,
                                        transform=transform)
    return 5000, train_dataset, val_dataset