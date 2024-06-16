import os
import timm.optim.optim_factory as optim_factory
import wandb
import argparse
import time
import random
import shutil
import warnings
import json
import yaml

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from torch.optim import lr_scheduler
from sklearn import metrics
from torch.autograd import Variable

from datasets.ff_all_semi import FaceForensics
from datasets.factory import create_data_transforms
from model.LVNet_semi import Two_Stream_Net
from loss.seg_loss import *
from utils.utils import *
from loss.patch_unsup_loss import *

torch.autograd.set_detect_anomaly(True)

def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()

    parser.add_argument("--opt", default='./config/FF++.yml', type=str, help="Path to option YMAL file.")
    
    parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--device', default=None, type=int,
                    help='GPU id to use.')

    parser.add_argument('--mixup', action="store_false",
                    help='using mixup augmentation.')
    
    parser.set_defaults(bottleneck=True)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    opt = yaml.safe_load(open(args.opt, 'r'))
    seed = opt["train"]["manual_seed"]
        

    if seed is not None:
        # random.seed(args.seed)
        if args.gpu is None:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    else:  
        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args, opt)
        

def main_worker(gpu, ngpus_per_node, args, opt):

    args.gpu = gpu

    config = wandb.config
    config = vars(args)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(args.gpu)

    elif args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device(args.local_rank)

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    else:
        pass
    
    # # wandb 
    # if args.gpu is not None or args.rank == 0:
    #     wandb.init(project='Face-Forgery-Detection', group='20230702', name='two-stream-xception', config=config)

    # create model
    print(f"Creating model: {opt['model']['baseline']}")
    model = Two_Stream_Net()
    model.to(args.device)

    if not opt['train']['resume'] == None:
        from_pretrained(model, opt)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                            output_device=args.local_rank, find_unused_parameters=True)
        param_groups = optim_factory.add_weight_decay(model.module, opt['train']['weight_decay'])
    else:
        param_groups = optim_factory.add_weight_decay(model, opt['train']['weight_decay'])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    segLoss = FSCELoss().to(args.device)

    optimizer = torch.optim.Adam(param_groups, lr=opt['train']['lr'], betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    cudnn.benchmark = True

    # Data loading code
    all_transform = create_data_transforms(opt)
    train_data = FaceForensics(opt, split='train', transforms=all_transform)
    val_data = FaceForensics(opt, split='val', transforms=all_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, rank=args.local_rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, rank=args.local_rank)

    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt['datasets']['train']['batch_size'], shuffle=(train_sampler is None),
        num_workers=opt['datasets']['n_workers'], sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=opt['datasets']['train']['batch_size'], shuffle=False,
        num_workers=opt['datasets']['n_workers'], sampler=val_sampler, drop_last=False)

    if (args.gpu is not None or args.local_rank == 0) and opt['train']['resume'] == None: 
        save_path = opt['train']['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        var_dict = vars(args)
        var_dict['optimizer'] = str(optimizer.__class__.__name__)
        var_dict['device'] = str(args.device)
        json_str = json.dumps(var_dict)
        with open(os.path.join(save_path, 'config.json'), 'w') as json_file:
            json_file.write(json_str)

    best = 0.0
    checkpoint_pth = './checkpoints'
    
    for epoch in range(opt['train']['start_epoch'], opt['train']['epoch']):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_acc, train_loss, train_celoss, train_segloss = train(train_loader, model, criterion, 
            segLoss, optimizer, epoch, args, opt)
        test_acc, test_loss, test_auc = validate(val_loader, model, criterion, epoch, args, opt)
        scheduler.step()

        is_best = test_auc > best
        best = max(test_auc, best)

        if args.gpu is not None or args.local_rank == 0:
            save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if args.gpu == None else model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, file=checkpoint_pth, epoch=epoch)
        
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']

            log_info = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                'train_celoss': train_celoss,
                'train_segloss': train_segloss,
                "test_acc": test_acc,
                "test_loss": test_loss,
                'test_auc': test_auc,
                'learning_rate': cur_lr
            }
            # wandb.log(log_info)


def train(train_loader, model, criterion, segLoss, optimizer, epoch, args, opt):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ce = AverageMeter('Loss_ce', ':.5f')
    losses_seg = AverageMeter('Loss_seg', ':.5f')
    losses = AverageMeter('Loss', ':.5f')
    acc = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')         
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_ce, losses_seg, losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, labels, masks, regions) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(args.device)
        labels = labels.to(args.device)
        masks = masks.to(args.device).float()
        regions = regions.to(args.device).float()

        masks[masks > 0], regions[regions > 0] = 1.0, 1.0
        # forward
        preds, psegs, pfeas, [masks, regions] = model(images, [masks, regions])

        loss_ce = criterion(preds, labels)
        # real image supervised loss
        if epoch < 2:
            loss_real_seg = segLoss(psegs[labels < 1], masks[labels < 1])
        else:
            loss_real_seg = 0.0
        # unsupervised loss
        loss_seg = sspsl_loss(psegs, pfeas, labels, regions)[0]

        # measure accuracy and record loss
        loss = loss_real_seg + loss_ce + loss_seg
        acc1 = accuracy(preds, labels, topk=(1,))[0]

        losses.update(loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses_seg.update(loss_seg.item(), images.size(0))
        acc.update(acc1, images.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (args.gpu is not None or args.local_rank == 0) and i % 20 == 0:
            progress.display(i)

    return acc.avg, losses.avg, losses_ce.avg, losses_seg.avg

@torch.no_grad()
def validate(val_loader, model, criterion, epoch, args, opt):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    y_trues = []
    y_preds = []
    for i, (images, labels, masks, _) in enumerate(val_loader):
        data_time.update(time.time() - end)
        
        images = images.to(args.device)
        labels = labels.to(args.device)

        # forward
        preds, psegs, _ = model(images, None)
            
        # measure accuracy and record loss
        loss = criterion(preds, labels)
        acc1 = accuracy(preds, labels, topk=(1,))[0]

        losses.update(loss.item(), images.size(0))
        acc.update(acc1, images.size(0))
        # top5.update(acc5, images.size(0))

        y_trues.extend(labels.cpu().numpy())
        prob = 1 - torch.softmax(preds, dim=1)[:, 0].cpu().numpy()
        y_preds.extend(prob)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (args.gpu is not None or args.local_rank == 0) and i % 20 == 0:
            progress.display(i)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    print(f' * Acc@1 {acc.avg:.3f}, auc {auc:.3f}')

    return acc.avg, losses.avg, auc


def save_checkpoint(state, is_best, epoch, file='checkpoint.pth.tar'):
    # if os.path.exists(filename):
    #     os.mkdir()
    filename = os.path.join(file, 'checkpoint-{:02d}.pth.tar'.format(epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(file, 'model_best.pth.tar'))


def from_pretrained(model, opt):
    state_dict = torch.load(opt['train']['resume'], map_location='cpu')
    model.load_state_dict(cleanup_state_dict(state_dict['state_dict']), strict=False)

    opt['train']['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
    opt['train']['start_epoch'] = state_dict['epoch']


if __name__ == '__main__':
    main()
