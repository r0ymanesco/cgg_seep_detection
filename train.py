import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network import UNet
from dataset import DataFolder
import torch.utils.data as data
from util import EarlyStopping, save_nets, save_predictions, load_best_weights
from train_options import parser


args = parser.parse_args()
print(args)

all_loader = data.DataLoader(
    dataset=DataFolder('dataset/all_images_256/', 'dataset/all_masks_256/', 'all'),
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=2
)

train_loader = data.DataLoader(
    dataset=DataFolder('dataset/train_images_256/', 'dataset/train_masks_256/', 'train'),
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=2
)

valid_loader = data.DataLoader(
    dataset=DataFolder('dataset/valid_images_256/', 'dataset/valid_masks_256/', 'validation'),
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=2
)

eval_loader = data.DataLoader(
    dataset=DataFolder('dataset/eval_images_256/', 'dataset/eval_masks_256/', 'evaluate'),
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=2
)

model = UNet(1, shrink=1).cuda()
nets = [model]
params = [{'params': net.parameters()} for net in nets]
solver = optim.Adam(params, lr=args.lr)

criterion = nn.CrossEntropyLoss()
es = EarlyStopping(min_delta=args.min_delta, patience=args.patience)

for epoch in range(1, args.epochs+1):

    train_loss = []
    valid_loss = []

    for batch_idx, (img, mask, _) in enumerate(train_loader):

        solver.zero_grad()

        img = img.cuda()
        mask = mask.cuda()

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

    print('[EPOCH {}/{}] Train Loss: {:.4f}; Valid Loss: {:.4f}'.format(
        epoch, args.epochs, np.mean(train_loss), np.mean(valid_loss)
    ))

    flag, best, bad_epochs = es.step(torch.Tensor([np.mean(valid_loss)]))
    if flag:
        print('Early stopping criterion met')
        break
    else:
        if bad_epochs == 0:
            save_nets(nets, 'model')
            print('Saving current best model')

        print('Current Valid loss: {:.4f}; Current best: {:.4f}; Bad epochs: {}'.format(
            np.mean(valid_loss), best.item(), bad_epochs
        ))

print('Training done... start evaluation')

with torch.no_grad():
    eval_loss = []
    for batch_idx, (img, mask, _) in enumerate(eval_loader):

        model = load_best_weights(model, 'model')

        img = img.cuda()
        mask = mask.cuda()

        pred = model(img)
        loss = criterion(pred, mask)

        eval_loss.append(loss.item())

        print('[EVALUATE {}/{}] Eval Loss: {:.4f}'.format(
            batch_idx+1, len(eval_loader), loss.item()
        ))

print('FINAL EVAL LOSS: {:.4f}'.format(np.mean(eval_loss)))

with torch.no_grad():
    all_loss = []
    for batch_idx, (img, mask, img_fns) in enumerate(all_loader):

        img = img.cuda()
        mask = mask.cuda()

        pred = model(img)
        loss = criterion(pred, mask)

        all_loss.append(loss.item())

        pred_mask = torch.argmax(F.softmax(pred, dim=1), dim=1)
        pred_mask = torch.chunk(pred_mask, chunks=args.eval_batch_size, dim=0)
        save_predictions(pred_mask, img_fns, 'output')

        print('[PREDICT {}/{}] Loss: {:.4f}'.format(
            batch_idx+1, len(all_loader), loss.item()
        ))

print('FINAL PREDICT LOSS: {:.4f}'.format(np.mean(all_loss)))
