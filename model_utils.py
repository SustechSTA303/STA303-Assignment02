from sklearn.metrics import roc_auc_score
import torch
import clip
import torch.nn.functional as F
import tqdm
import numpy as np
from tqdm import tqdm

def prompt_encode(prompt, class_names):
    """
    Args:
        prompt (str): the text prefix before the class

    Returns:
        text_inputs(torch.Tensor): [the prefix + class name]

    """
    text_inputs = torch.cat([clip.tokenize(prompt + " " + c) for c in class_names])
    
    return text_inputs

def model_inference(model, image, text_inputs):
    """
    Args:
        model (torch.nn.Module): CLIP model
        image (torch.Tensor): a list of image tensor
        text_inputs (torch.Tensor): a list of text tensor
    Returns:
        logits (torch.Tensor): logits of the image and text pair (one row per image)
    """
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text_inputs.to(image.device))
        logits = logits_per_image

    return logits

def calc_msp(logics):
    """
    Args:
        logics (torch.Tensor): logits of the image and text pair (one row per image)
    Returns:
        msp (torch.Tensor): maximum softmax probability
    """
    probs = F.softmax(logics, dim=-1)
    msp, _ = torch.max(probs, dim=-1)
    return msp


def calc_msp_threshold(in_dataloader, model, prompt, class_names, percentage, device):
    """
    Args:
        in_dataloader (torch.utils.data.DataLoader): in-distribution test set dataloader
        model (torch.nn.Module): CLIP model
        prompt (str): the text prefix before the class
        class_names (list): class names
        percentage (int): the percentage of the in-distribution test set
        device (torch.device): cpu or gpu
    Returns:
        threshold (float): MSP threshold
        in_acc (float): classification accuracy of imagenet
    """
    in_logits = []
    in_accs = []
    for batch in tqdm(in_dataloader, mininterval=10):
        image, target = batch
        image = image.to(device)
        text_inputs = prompt_encode(prompt, class_names)
        logits = model_inference(model, image, text_inputs)
        in_logits.append(logits)
        in_accs.append(torch.argmax(logits, dim=1) == target.to(device))
    in_logits = torch.cat(in_logits)
    in_accs = torch.cat(in_accs)
    in_msp = calc_msp(in_logits)
    in_msp = in_msp.cpu().numpy()
    threshold = np.percentile(in_msp, percentage)
    print(f"MSP {percentage}% threhold:", threshold)
    in_acc = torch.mean(in_accs.float()).cpu()
    print("Classification accuracy of imagenet:", float(in_acc))
    return threshold, in_acc

def eval_msp(ood_dataloader, model, prompt, class_names, threshold, device, dataset_name):
    """
    Args:
        ood_dataloader (torch.utils.data.DataLoader): out-of-distribution test set dataloader
        model (torch.nn.Module): CLIP model
        prompt (str): the text prefix before the class
        class_names (list): class names
        threshold (float): MSP threshold
        device (torch.device): cpu or gpu
    Returns:
        ood_acc (float): classification accuracy of half out-of-distribution test set and half in-distribution test set
    """
    ood_logits = []
    for batch in tqdm(ood_dataloader, mininterval=10):
        image, _ = batch
        image = image.to(device)
        text_inputs = prompt_encode(prompt, class_names)
        logits = model_inference(model, image, text_inputs)
        ood_logits.append(logits)
    ood_logits = torch.cat(ood_logits)
    ood_msp = calc_msp(ood_logits)
    ood_msp = ood_msp.cpu().numpy()
    ood_pred = ood_msp < threshold
    ood_acc = np.mean(ood_pred)
    print(f"Accuracy of {dataset_name}:", ood_acc)
    return  ood_acc