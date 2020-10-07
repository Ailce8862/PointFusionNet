import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models import PFNet_MSG_Cls as Fusion_MSG_Cls
from data import ModelNet40Cls
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
import time
import logging
import sklearn.metrics as metrics

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='PointFsuionNet Classification Training')
parser.add_argument('--config', default='cfgs/config_msg_cls.yaml', type=str)


def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:' % (k), v)
    print("\n**************************\n")

    try:
        os.makedirs(args.save_path)
    except OSError:
        pass

    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    train_dataset = ModelNet40Cls(num_points=args.num_points, root=args.data_root, transforms=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True
    )

    test_dataset = ModelNet40Cls(num_points=args.num_points, root=args.data_root, transforms=test_transforms,
                                 train=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True
    )

    model = Fusion_MSG_Cls(num_classes=args.num_classes, input_channels=args.input_channels,
                           relation_prior=args.relation_prior, use_xyz=True)
    model.cuda()
    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    print('\nnum_parameters =%d' % (num_parameters))
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    # optimizer=optim.Adam(model.parameters(),lr=args.base_lr,betas=(0.9,0.99),weight_decay=1e-8)
    # optimizer=torch.optim.SGD(model.parameters(),lr=args.base_lr,momentum=0.9)
    # weight_p, bias_p = [], []
    # for name, p in model.named_parameters():
    #     print(name)
    #     if 'bias' in name:
    #         bias_p += [p]
    #     else:
    #         weight_p += [p]
    # for layer,para in model.state_dict().items():
    #     print(layer)
    #     print(para)

    # # 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias
    # optimizer=torch.optim.SGD(
    #     [{'params': weight_p, 'weight_decay': 1e-5},
    #     {'params': bias_p, 'weight_decay': 0}], lr=args.base_lr, momentum=0.9)

    lr_lbmd = lambda e: max(args.lr_decay ** (e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay ** (e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    num_batch = len(train_dataset) / args.batch_size
    # print('train_dataset is len and train_dataset')
    # print(len(train_dataset))

    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)


def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()  # initialize augmentation
    PointcloudJitter = d_utils.PointcloudJitter()
    PointcloudRandomInputDropout = d_utils.PointcloudRandomInputDropout()

    global g_acc
    g_acc = 0.9100  # only save the model whose acc > 0.91# nn.Dropout(p=0.5),
    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            # 添加
            # print(i)
            # print(data)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch - 1)
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)

            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, 1200)  # (B, npoint)
            # print(fps_idx[1])
            fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                              2).contiguous()  # (B, N, 3)

            # augmentation
            points.data = PointcloudScaleAndTranslate(points.data)
            # points.data=PointcloudJitter(points.data)
            # points.data=PointcloudRandomInputDropout(points.data)train_msg_cls.py:153

            optimizer.zero_grad()

            # print(points.size())[B,1024,3]
            pred = model(points)
            target = target.view(-1)
            # print(target)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' % (
                    epoch + 1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1

            # validation in between an epoch
            if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                validate(test_dataloader, model, criterion, args, batch_count)


def validate(test_dataloader, model, criterion, args, iter):
    global g_acc
    model.eval()
    losses, preds, labels = [], [], []
    for j, data in enumerate(test_dataloader, 0):
        points, target = data
        points, target = points.cuda(), target.cuda()
        points, target = Variable(points, volatile=True), Variable(target, volatile=True)

        # fastest point sampling
        fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
        # fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                          2).contiguous()

        pred = model(points)
        target = target.view(-1)
        loss = criterion(pred, target)
        losses.append(loss.data.clone())
        _, pred_choice = torch.max(pred.data, -1)

        preds.append(pred_choice)
        labels.append(target.data)

    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    acc = (preds == labels).sum().item() / labels.numel()
    # acc = metrics.accuracy_score(labels, preds)
    # avg_per_class_acc = metrics.balanced_accuracy_score(labels, preds)
    print('\nval loss: %0.6f \t acc: %0.6f \n' % (np.array(losses).mean(), acc))
    if acc > g_acc:
        g_acc = acc
        torch.save(model.state_dict(), '%s/fusion_all_cls_msg_iter_%d_acc_%0.6f.pth' % (args.save_path, iter, acc))
        print('Saved successfully')
    model.train()


if __name__ == "__main__":
    start = time.time()
    main()
    torch.cuda.synchronize()
    end = time.time()
    print('\ntime =%f' % ((end - start) / 3600))