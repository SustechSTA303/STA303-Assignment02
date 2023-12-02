Models: For CLIP, I choose the CLIP model with RN50x64 as the image encoder (However, on CIFAR100, I use Vit-L/14 rather than RN50x64 because it demonstrates significantly better performance).
For the baseline model, I employ ResNet50 which has been pretrained.

Datasets(Scenario): I choose Mnist, Oxford-IIITPets and CIFAR100 as benchmark tasks. 

Metric: I compute and compare the zero-shot accuracy of Clip and fine-tuning accuracy of ResNet50.

Mnist.ipynb Pet.ipynb Cifar100.ipynb are the files to evaluate the performance of Clip on different datasets.
funct.py contains some functions used in fine-tuning or evaluating.

