# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from clip import clip
import torchvision.models
from torchvision import transforms
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Hyperparameters
SEED = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
STEP=5
GAMMA=0.5

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Model
def mod(NUM_CLASS):
    CLIP_RN50, CLIP_RN50_preprocess = clip.load(name='RN50', device=device, download_root='/shareddata/clip/')
    CLIP_ViT_B_32, CLIP_ViT_B_32_preprocess = clip.load(name="ViT-B/32", device=device, download_root='/shareddata/clip/')
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, NUM_CLASS)
    models = {
        'CLIP_RN50':  {'model': CLIP_RN50, 'preprocess': CLIP_RN50_preprocess},
        'CLIP_ViT_B_32': {'model': CLIP_ViT_B_32, 'preprocess': CLIP_ViT_B_32_preprocess},
        'RN50': {'model': resnet50},
    }
    return models

# Data
def transform_test_with_channel_check(image):
    channels = image.getbands()
    if len(channels) == 1:
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image)

prompt = 'a photo of a'
def prompt_encode(prompt, class_names):
    text_inputs = torch.cat([clip.tokenize(f'{prompt} {c}') for c in class_names]).to(device)
    return text_inputs

def model_inference(model, image, text_inputs):
    image = image.to(device)
    model = model.to(device)
    text_inputs = text_inputs.to(device)
    image_features = model.encode_image(image)
    logits, _ = model(image, text_inputs)
    return logits

def zero_shot(test_set, dataset_name, NUM_CLASS):
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                                  shuffle=False, num_workers=2)
    class_names = test_set.classes
    result = {}
    models = mod(NUM_CLASS)
    for models_name, models_info in models.items():
        model = models_info['model']
        model.to(device)
        result_data = {
            'testing_loss': 0,
            'testing_acc': 0
        }
        testing_loss = 0
        testing_acc = 0
        with torch.no_grad():
            model.eval()
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                text_inputs = prompt_encode(prompt, class_names)
                if(models_name == 'CLIP_RN50' or models_name == 'CLIP_ViT_B_32'):
                    logits = model_inference(model, images, text_inputs)
                else:
                    logits = model(images)
                predictions = torch.argmax(logits, dim=1)
                accuracy = torch.sum(predictions == labels).item()
                loss = criterion(logits, labels)

                testing_acc += accuracy
                testing_loss += loss.item()

            avg_acc = testing_acc / len(test_set)
            avg_loss = testing_loss  / len(test_set)

            result_data['testing_loss']=avg_loss
            result_data['testing_acc']=avg_acc
            print(f"the zero-shot performance on {dataset_name} is {avg_acc*100:.2f}%, visual encoder is {models_name}.")
        result[models_name] = result_data
    print(result)
    return result

def get_features(dataset,model):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            images = images.to(device)
            labels = labels.to(device)
            model = model.to(device)
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def linear_probe(train_set, test_set, dataset_name, NUM_CLASS):
    result = {}
    models = mod(NUM_CLASS)
    for models_name, models_info in models.items():
        if models_name == 'RN50':
            continue
        result_data = {}
        model = models_info['model']
        model = model.to(device)
        train_features, train_labels = get_features(train_set, model)
        test_features, test_labels = get_features(test_set, model)
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(test_features)
        accuracy = np.mean((test_labels == predictions).astype(float))
        print(f"the linear_probe performance on {dataset_name} is {accuracy:.3f}%, visual encoder is {models_name}.")
        result_data['testing_acc'] = accuracy
        result[models_name] = result_data
    print(result)
    return result

def prob_images(test_set, NUM_CLASS, class_names,dataset_name):
    indices = []
    for i in range(NUM_CLASS):
        indices.extend(np.where(np.array(test_set.targets) == i)[0][:1])
    few_shot_dataset = torch.utils.data.Subset(test_set, indices)
    few_shot_dataloader = torch.utils.data.DataLoader(few_shot_dataset, batch_size=128, shuffle=True, num_workers=2)
    for i, (input, label) in enumerate(few_shot_dataloader, 0):
        input, label = input.to(device), label.to(device)
    models = mod(NUM_CLASS)
    for models_name, models_info in models.items():
        model = models_info['model']
        model = model.to(device)
        with torch.no_grad():
            model.eval()
            for images, labels in few_shot_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                text_inputs = prompt_encode(prompt, class_names)
                if(models_name == 'CLIP_RN50' or models_name == 'CLIP_ViT_B_32'):
                    logits = model_inference(model, images, text_inputs)
                else:
                    logits = model(images)
                prob = F.softmax(logits, dim=-1)
                prob = prob.to('cpu').float()
                topk_values, topk_indices = torch.topk(prob, 3, dim=1)
                sns.set(style="whitegrid")
                fig, axes = plt.subplots(5, 4, figsize=(12, 12))
                for index in range(10):
                    orgin = index * 2
                    pred = index * 2 + 1
                    axes[orgin // 4, orgin % 4].imshow(images[index].cpu().numpy().transpose((1, 2, 0)))
                    axes[orgin // 4, orgin % 4].set_title(f"{class_names[labels[index]]}")
                    axes[orgin // 4, orgin % 4].axis('off')

                    values = topk_values[index].numpy()
                    indices = topk_indices[index].numpy()
                    sns.barplot(x=[f"{class_names[idx]}" for idx in indices], y=values,
                                ax=axes[pred // 4, pred % 4])
                plt.tight_layout()
                plt.savefig(f'images/{dataset_name}_{models_name}.png')
                plt.show()

def zero_linear_image(datasets_zero_shot,datasets_linear_probe):
    plt.figure(figsize=(8, 6))
    for dataset, models in datasets_zero_shot.items():
        model_names = list(models.keys())
        accuracies = [models[model]['testing_acc'] for model in model_names]
        plt.plot(model_names, accuracies, marker='o', label=dataset)
        plt.legend()
        plt.title('Model Accuracy Comparison across Datasets')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
    for dataset, models in datasets_linear_probe.items():
        model_names = list(models.keys())
        accuracies = [models[model]['testing_acc'] for model in model_names]
        plt.plot(model_names, accuracies, marker='o', label=dataset)
        plt.legend()
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
    plt.savefig('images/zero_linear_image')


if __name__ == '__main__':
    # Zero-shot & Linear-probe
    datasets_zero_shot = {}
    datasets_linear_probe = {}
    ## MINST
    minst_train = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform_test_with_channel_check)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform_test_with_channel_check)
    datasets_zero_shot['MNIST(Zero Shot)'] = zero_shot(mnist_test, 'MNIST',len(mnist_test.classes))
    datasets_linear_probe['MNIST(Linear Probe)'] = linear_probe(minst_train, mnist_test, 'MNIST', len(mnist_test.classes))

    ## FashionMNIST
    fashion_mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                            download=True, transform=transform_test_with_channel_check)
    fashion_mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                           download=True, transform=transform_test_with_channel_check)
    datasets_zero_shot['FashionMNIST(Zero Shot)'] = zero_shot(fashion_mnist_test, 'FashionMNIST',len(fashion_mnist_test.classes))
    datasets_linear_probe['FashionMNIST(Linear Probe)'] = linear_probe(fashion_mnist_train, fashion_mnist_test, 'FashionMNIST', len(fashion_mnist_test.classes))


    ## Food101
    food101_train = torchvision.datasets.Food101(root='./data', split='train',
                                            download=True, transform=transform_test_with_channel_check)
    food101_test = torchvision.datasets.Food101(root='./data', split='test',
                                            download=True, transform=transform_test_with_channel_check)
    datasets_zero_shot['Food101(Zero Shot)'] = zero_shot(food101_test, 'Food101',len(food101_test.classes))
    datasets_linear_probe['Food101(Linear Probe)'] = linear_probe(food101_train, food101_test, 'Food101', len(food101_test.classes))
    print(datasets_zero_shot)
    print(datasets_linear_probe)
    zero_linear_image(datasets_zero_shot,datasets_linear_probe)

    # 画概率图
    prob_images(mnist_test, len(mnist_test.classes), mnist_test.classes, 'MNIST')
    prob_images(fashion_mnist_test, len(fashion_mnist_test.classes), fashion_mnist_test.classes, 'FashionMNIST')



