import numpy as np 
import os 
import time 
import ipdb

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable
import torch.nn.functional as F 
from network import UNet
from dataset import DataFolder
import torch.utils.data as data
from util import EarlyStopping, save_nets, save_predictions

train_batch_size = 16
eval_batch_size = 16
epochs = 100
lr = 0.001

all_loader = data.DataLoader(
    dataset=DataFolder('dataset/all_images_256/', 'dataset/all_masks_256/', 'all'),
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=2
)

train_loader = data.DataLoader(
    dataset=DataFolder('dataset/train_images_256/', 'dataset/train_masks_256/', 'train'),
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=2
)

valid_loader = data.DataLoader(
    dataset=DataFolder('dataset/valid_images_256/', 'dataset/valid_masks_256/', 'validation'),
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=2
)

eval_loader = data.DataLoader(
    dataset=DataFolder('dataset/eval_images_256/', 'dataset/eval_masks_256/', 'evaluate'),
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=2
)

if not os.path.exists('model'):
    print('Creating model directory: {}'.format('model'))
    os.makedirs('model')

model = UNet(1, shrink=1).cuda()
nets = [model]
params = [{'params': net.parameters()} for net in nets]
solver = optim.Adam(params, lr=lr)

criterion = nn.CrossEntropyLoss()
es = EarlyStopping(min_delta=0.001, patience=10)

for epoch in range(1, epochs+1):

    train_loss = []
    valid_loss = []

    for batch_idx, (img, mask, _) in enumerate(train_loader):

        solver.zero_grad()

        img = img.cuda()
        mask = mask.cuda()

        # ipdb.set_trace() 

        pred = model(img)
        loss = criterion(pred, mask)

        loss.backward()
        solver.step()

        train_loss.append(loss.item())

    with torch.no_grad():
        for batch_idx, (img, mask, _) in enumerate(valid_loader):

            img = img.cuda()
            mask = mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            valid_loss.append(loss.item())

    print('[EPOCH {}/{}] Train Loss: {}; Valid Loss: {}'.format(
        epoch, epochs, np.mean(train_loss), np.mean(valid_loss)
    ))

    if epoch % 10 == 0:
        save_nets(nets, 'model')

    if es.step(torch.Tensor([np.mean(valid_loss)])):
        save_nets(nets, 'model')
        print('Early stopping criterion met')
        break

    break

save_nets(nets, 'model')
print('Training done... start evaluation')

with torch.no_grad():
    eval_loss = []
    for batch_idx, (img, mask, _) in enumerate(eval_loader):

        img = img.cuda()
        mask = mask.cuda()

        pred = model(img)
        loss = criterion(pred, mask)

        eval_loss.append(loss.item())

        print('[EVALUATE {}/{}] Eval Loss: {}'.format(
            batch_idx, len(eval_loader), loss.item()
        ))

print('FINAL EVAL LOSS: {}'.format(np.mean(eval_loss)))


with torch.no_grad():
    all_loss = []
    for batch_idx, (img, mask, img_fns) in enumerate(all_loader):

        img = img.cuda()
        mask = mask.cuda()

        pred = model(img)
        loss = criterion(pred, mask)

        all_loss.append(loss.item())

        pred_mask = torch.argmax(F.softmax(pred, dim=1), dim=1)
        pred_mask = torch.chunk(pred_mask, chunks=eval_batch_size, dim=0)
        save_predictions(pred_mask, img_fns, 'output')

        print('[PREDICT {}/{}] Loss: {}'.format(
            batch_idx, len(all_loader), loss.item()
        ))

print('FINAL PREDICT LOSS: {}'.format(np.mean(all_loss)))


    



