import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import tqdm
import clip
import torch
import torch.nn as nn
import torch.optim as optim

def model_inference(model,image,text_inputs):
    logits,logit=model(image,text_inputs)
    return logits


def clip_testing(model,preprocess,clip_test,device,text_inputs):
    corrects=0
    model.eval()

    for i in tqdm.tqdm(range((len(clip_test)))):
        image,target=clip_test[i]
        image_input=preprocess(image).unsqueeze(0).to(device)
        logits=model_inference(model,image_input,text_inputs)
        prob=logits.softmax(dim=-1)
        _,preds=torch.max(prob,1)
        if(preds==target):
            corrects+=1
        
        if(i%10==0):
            torch.cuda.empty_cache()
            
    accuracy=corrects/len(clip_test)
    return accuracy


def resnet_training (resnet50,criterion,optimizer,train_dataloader,device):
    resnet50.train()
    for batchidx,(image,target) in enumerate(tqdm.tqdm(train_dataloader)):
        image=image.to(device)
        target=target.to(device)
        outputs=resnet50(image)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        
        
def resnet_testing(resnet50,test_dataloader,device):
    resnet_corrects=0
    resnet50.eval()
    for batchidx,(image,target) in enumerate(tqdm.tqdm(test_dataloader)):
        image=image.to(device)
        target=target.to(device)
        outputs=resnet50(image)
        _, preds = torch.max(outputs, 1)
        resnet_corrects+=torch.sum(preds == target.data)
        torch.cuda.empty_cache()
    
    corrects=resnet_corrects.item()
    return corrects