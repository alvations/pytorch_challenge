# Imports here
import torch
import torchvision

from tqdm import tqdm
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.utils.model_zoo as model_zoo
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

import json

from CLR_preview import CyclicLR
from adamW import AdamW

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# TODO: Define your transforms for the training and validation sets
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.RandomRotation(90),
   transforms.CenterCrop(224),
   transforms.ToTensor(), 
])

preprocess_validate = transforms.Compose([
                        transforms.Resize(256), #resize shorter side to 256
                        transforms.CenterCrop(224), #center crop
                        #transforms.TenCrop(224),
                        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor(crop) for crop in crops])),
                        #transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
                        transforms.ToTensor(), 
                       ])


# TODO: Load the datasets with ImageFolder
image_datasets = {'train': ImageFolder(root=train_dir, transform=preprocess),
                  'valid': ImageFolder(root=valid_dir, transform=preprocess_validate)}


batchsz = 50 
# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=int(1000), shuffle=True, pin_memory=True),
               'valid': DataLoader(image_datasets['valid'], batch_size=int(1000), shuffle=True, pin_memory=True)}

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

num_classes = len(cat_to_name)

resnet = models.resnet50(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

num_feats = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_feats, num_classes)

resnet = nn.DataParallel(resnet, device_ids=[0,1,2,3]).to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

optimizer = AdamW(resnet.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay = 0.1)

clr_stepsize = (num_classes*50//int(batchsz))*4
clr_wrapper = CyclicLR(optimizer, step_size=clr_stepsize)

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloaders['train'])):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        clr_wrapper.batch_step()

        # print statistics
        running_loss += loss.item()
        if i % 1 == 0:    # print every mini-batches
            lrs = [p['lr'] for p in optimizer.param_groups]
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100), lrs, flush=True)
            running_loss = 0.0

    valid_loss = 0.0
    for i, data in tqdm(enumerate(dataloaders['valid'])):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = resnet(inputs)    
        loss = criterion(outputs, labels)
        valid_loss += loss.item()

    print('Epoch %d valid loss: %.3f' % (epoch + 1, valid_loss / 100), flush=True)
            
    torch.save(resnet.state_dict(), 'models/{}.pth'.format(epoch))


