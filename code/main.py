import argparse
import random
import numpy as np
import torch 
import torchvision.transforms as transforms

from Data.create_dataset import get_data
from Model.model import CLIP
from ConformalPrediction.create_conformal import create_conformal
from Utils.utils import get_prob_targets, solve_metric, get_logits_targets, write


def main() -> None:
    parser = argparse.ArgumentParser(description='Conformal Prediction Model on Imagenet')
    parser.add_argument('--seed', type=int, metavar='SEED', help='random seed', default=114514)
    parser.add_argument('--num_workers', type=int, metavar='NW', help='number of workers', default=0)
    parser.add_argument('--model', type=str, metavar='MDL', help='which model to use', default='clip')
    parser.add_argument('--backbone', type=str, metavar='BB', help='Image Backbone', default='ViT-B/16')
    parser.add_argument('--batch_size', type=int, metavar='BSZ', help='batch size', default=128)
    parser.add_argument('--conformal', type=str, metavar='CFM', help='the method of conformal prediction', default='APS')
    parser.add_argument('--alpha', type=float, metavar='ALPHA', help='confidence level', default=0.05)
    parser.add_argument('--randomized', type=bool, metavar='RDM', help='whether to use randomized process', default=True)
    parser.add_argument('--allow_zero_sets', type=bool, metavar='AZS', help='whether you allow zero sets', default=True)
    parser.add_argument('--num_calib', type=int, metavar='NC', help='number of calibration dataset size', default=1000)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    # np.random.seed(seed=args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)

    #------------ load model ------------
    prompt = "This is a"
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    transform_cifar10_test = transforms.Compose([
                             transforms.Resize(size=224),
                             transforms.CenterCrop(size=(224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])
    model = CLIP(args.backbone, prompt, class_names)

    #------------ load dataset ------------
    num_calib, train_dataset, val_dataset = get_data("Cifar10", transform_cifar10_test)
    num_calib = args.num_calib
    dataset_length = len(val_dataset)
    calib_data, val_data = torch.utils.data.random_split(val_dataset, [num_calib, dataset_length-num_calib])
    calib_loader = torch.utils.data.DataLoader(calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    #------------ load conformal prediction ------------
    Conformal = create_conformal(args.conformal, args.alpha, args.randomized, args.allow_zero_sets)

    # calculate threshold
    calib_probs_loader = get_prob_targets(model, calib_loader, 10)

    Conformal.solve_qhat(calib_probs_loader)

    # evaluation
    val_probs_loader = get_prob_targets(model, val_loader, 10)
    top1, top5, coverage, size = solve_metric(Conformal, val_probs_loader)
    write(args.backbone, top1, top5, coverage, size, num_calib)