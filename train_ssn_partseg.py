import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models import PFNet_SSN_Seg as Fusion_SSN_Seg
from data import ShapeNetPart
import utils.pytorch_utils as pt_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='PointFsuionNet Shape Part Segmentation Training')
parser.add_argument('--config', default='cfgs/config_ssn_partseg.yaml', type=str)


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

    train_dataset = ShapeNetPart(root=args.data_root, num_points=args.num_points, split='trainval', normalize=True,
                                 transforms=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True
    )

    global test_dataset
    test_dataset = ShapeNetPart(root=args.data_root, num_points=args.num_points, split='test', normalize=True,
                                transforms=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True
    )
    # seg_classes = test_dataset.seg_classes
    # shape_ious = {cat: [] for cat in seg_classes.keys()}
    # print(shape_ious)
    # {'Pistol': [], 'Earphone': [], 'Car': [], 'Laptop': [], 'Rocket': [], 'Mug': [], 'Guitar': [], 'Knife': [], 'Chair': [], 'Table': [], 'Skateboard': [], 'Cap': [], 'Lamp': [], 'Motorbike': [], 'Bag': [], 'Airplane': []}
    # print(len(test_dataset.seg_classes))
    # print(test_dataset.seg_classes.keys())
    # ['Mug', 'Airplane', 'Earphone', 'Table', 'Skateboard', 'Motorbike', 'Lamp', 'Laptop', 'Chair', 'Car', 'Guitar', 'Pistol', 'Knife', 'Cap', 'Rocket', 'Bag']
    # seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    # for cat in seg_classes.keys():
    #     for label in seg_classes[cat]:
    #         seg_label_to_cat[label] = cat
    # 将各个部分的类别标签赋予物品种类名字
    #         print(seg_classes[cat])
    #         print(cat)
    #         print(label)
    #         print(seg_label_to_cat)

    model = Fusion_SSN_Seg(num_classes=args.num_classes, input_channels=args.input_channels,
                      relation_prior=args.relation_prior, use_xyz=True)
    model.cuda()
    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    print('\nnum_parameters =%d' % (num_parameters))
    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay ** (e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay ** (e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    num_batch = len(train_dataset) / args.batch_size
    # print(len(train_dataset))
    # print(num_batch)

    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)


def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()  # initialize augmentation
    global Class_mIoU, Inst_mIoU
    Class_mIoU, Inst_mIoU = 0.83, 0.85
    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch - 1)

            points, target, cls = data
            # print(points.size()) [B,2048,3]
            # print(target.size()) [B,2048]
            # print(cls.size()) [B,1]
            # print(len(cls))  [B]
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            # augmentation
            points.data = PointcloudScaleAndTranslate(points.data)

            optimizer.zero_grad()

            batch_one_hot_cls = np.zeros((len(cls), 16))  # 16 object classes
            # print(batch_one_hot_cls)
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
                # print(batch_one_hot_cls)将对应的种类所在的行改为1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
            # print(batch_one_hot_cls.size()) [14,16] 每个batch有一个对应一个种类
            batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

            pred = model(points, batch_one_hot_cls)
            # print(pred.size()) [14,2048,50]
            pred = pred.view(-1, args.num_classes)#view函数的-1参数的作用在于基于另一参数，自动计算该维度的大小;第二个参数用于计算
            # print(pred)
            # print(pred.size()) [28672,50] B*2048=28672
            target = target.view(-1, 1)[:, 0]
            # print(target.view(-1, 1))
            # print(target)
            # print(target.size()) [28672]
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
        # validate(test_dataloader, model, criterion, args, batch_count)



def validate(test_dataloader, model, criterion, args, iter):
    global Class_mIoU, Inst_mIoU, test_dataset
    model.eval()

    seg_classes = test_dataset.seg_classes
    # print(seg_classes) {'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Lamp': [24, 25, 26, 27], 'Guitar': [19, 20, 21], 'Laptop': [28, 29], 'Chair': [12, 13, 14, 15], 'Rocket': [41, 42, 43], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49], 'Knife': [22, 23], 'Cap': [6, 7], 'Car': [8, 9, 10, 11], 'Mug': [36, 37], 'Bag': [4, 5], 'Earphone': [16, 17, 18]}
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    # print(shape_ious) {'Airplane': [], 'Chair': [], 'Pistol': [], 'Laptop': [], 'Knife': [], 'Motorbike': [], 'Car': [], 'Lamp': [], 'Table': [], 'Cap': [], 'Skateboard': [], 'Rocket': [], 'Guitar': [], 'Mug': [], 'Bag': [], 'Earphone': []}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat#将各个部分的类别标签赋予物品种类名字
    # print(seg_label_to_cat) {0:Airplane, 1:Airplane, ...49:Table}

    losses = []
    for _, data in enumerate(test_dataloader, 0):
        points, target, cls = data
        # print(points)#[B,2048,3]
        # print(target) [B,2048]
        # print(cls) [B,1] 其值为0～15
        points, target = Variable(points, volatile=True), Variable(target, volatile=True)
        points, target = points.cuda(), target.cuda()

        batch_one_hot_cls = np.zeros((len(cls), 16))  # 16 object classes
        # print(batch_one_hot_cls) 全0,[B,16]
        for b in range(len(cls)):
            batch_one_hot_cls[b, int(cls[b])] = 1
        batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
        batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

        pred = model(points, batch_one_hot_cls)
        # print(pred)[B,2048,50]
        loss = criterion(pred.view(-1, args.num_classes), target.view(-1, 1)[:, 0])
        losses.append(loss.data.clone())
        pred = pred.data.cpu()
        target = target.data.cpu()
        pred_val = torch.zeros(len(cls), args.num_points).type(torch.LongTensor)
        # print(pred_val) [B,2048],最后一个B会变
        # pred to the groundtruth classes (selected by seg_classes[cat])
        for b in range(len(cls)):
            # print(target[b, 0])一个数0～49
            cat = seg_label_to_cat[target[b, 0]]
            # print(cat) 物品种类的名字
            logits = pred[b, :, :]  # (num_points, num_classes)
            # print(pred) [2048,50]
            # print(logits) [2048,50] 小数，非整数
            pred_val[b, :] = logits[:, seg_classes[cat]].max(1)[1] + seg_classes[cat][0]
            # print(logits[:, seg_classes[cat]])
            # [2048,每个物品分割的部分数] eg:'Airplane': [0, 1, 2, 3]=>[2048,4]

            # print(logits[:, seg_classes[cat]].max(1)[1])
            # [2048] 返回部分中概率最大的部分的索引 max(0)返回该矩阵中每一列的最大值;max(1)返回该矩阵中每一行的最大值

            # print(seg_classes[cat][0])
            # 由物品的类别确定分割的部分的第一个标签 可取：16,30,41,8,28,6,44,36,19,4,24,47,0,38,12,22

            # print(pred_val[b, :])
            # [2048]根据物品的类别和第一个标签，确认其实际标签0～49,logits的.max(1)[1]转变而来

        for b in range(len(cls)):
            # print(points[b, :,:])[2048,3]
            segp = pred_val[b, :]
            # print(segp)[2048]取值为0～49预测的标签
            segl = target[b, :]
            # print(segl) [2048]取值为0～49 真值的标签
            cat = seg_label_to_cat[segl[0]]
            # print(cat)输出物品的种类的名字
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            # print(part_ious)一个list，长度由物体所包含的部分决定
            # print(seg_classes[cat])一个list，物体所包含的部分标签0~49
            for l in seg_classes[cat]:
                if torch.sum((segl == l) | (segp == l)) == 0:
                    # part is not present in this shape
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = torch.sum((segl == l) & (segp  == l)) / float(
                        torch.sum((segl == l) | (segp == l)))
            # print(part_ious[l - seg_classes[cat][0]])取1.0或者else计算的结果；该计算结果为
            shape_ious[cat].append(np.mean(part_ious))
            # print(shape_ious[cat])
            # 将结果的均值存入对应的cat的list

    instance_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            instance_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_class_ious = np.mean(list(shape_ious.values()))

    for cat in sorted(shape_ious.keys()):
        print('****** %s: %0.6f' % (cat, shape_ious[cat]))
    print('************ Test Loss: %0.6f' % (np.array(losses).mean()))
    print('************ Class_mIoU: %0.6f' % (mean_class_ious))
    print('************ Instance_mIoU: %0.6f' % (np.mean(instance_ious)))

    if mean_class_ious > Class_mIoU or np.mean(instance_ious) > Inst_mIoU:
        if mean_class_ious > Class_mIoU:
            Class_mIoU = mean_class_ious
        if np.mean(instance_ious) > Inst_mIoU:
            Inst_mIoU = np.mean(instance_ious)
        torch.save(model.state_dict(), '%s/seg_ssn_iter_%d_ins_%0.6f_cls_%0.6f.pth' % (
        args.save_path, iter, np.mean(instance_ious), mean_class_ious))
        print('Saved successfully')
        torch.cuda.empty_cache()  # 4.2
    model.train()


if __name__ == "__main__":
    start = time.time()
    main()
    torch.cuda.synchronize()
    end = time.time()
    print('\ntime =%f' % ((end - start)/3600))