import torch
import clip
import numpy as np
import torch.nn.functional as F
import msp

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


