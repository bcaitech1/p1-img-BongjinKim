import argparse
import glob
import json
import os
import random
import re
import math
from importlib import import_module
from pathlib import Path
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import torch
from f1 import F1Score
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import wandb
from adamp import AdamP, SGDP
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict
from tqdm import tqdm

from dataset import MaskBaseDataset
from loss import create_criterion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_sweep_config(args):
    sweep_config = {
        'method':'random'
    }
    metric = {
        'name':'val_acc',
        'goal':'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict={
        'optimizer':{
            'value' : 'SGDP'
        },
        'criterion':{
            'value' : 'cross_entropy'
        },
        'epochs':{
            'value':1
        },
        'learning_rate':{
            'distribution':'uniform',
            'min':0.003,
            'max':0.0031
        },
        'batch_size':{
            'distribution':'q_log_uniform',
            'q':1,
            'min':math.log(32),
            'max':math.log(33)
        },
        'num_sample':{
            'value' : 3
        },
        'scheduler':{
            'value' : 'cosinelr'
        },
        'data_dir':{
            'value':'/opt/ml/input/data/train/images'
        },
        'model_dir':{
            'value':'/opt/ml/model'
        },
        'file':{
            'value':args.file
        },
        'seed':{
            'value':args.seed
        },
        'dataset':{
            'value':args.dataset
        },
        'augmentation':{
            'value':args.augmentation
        },
        'resize':{
            'value':args.resize
        },
        'valid_batch_size':{
            'value':args.valid_batch_size
        },
        'model_name':{
            'value':args.model_name
        },
        'log_interval':{
            'value':args.log_interval
        },
        'accumulation_steps':{
            'value':2
        },
        'focal_gamma':{
            'value':2
        }
        
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{config.file}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
    
def weighted_sampler(train_set, num):
    classes = []
    for img, label in train_set:
        classes.append(label)
    _, counts=np.unique(np.array(classes), return_counts=True)
    class_weights = [sum(counts) // c for c in counts]
    example_weights = [class_weights[i] for i in classes]
    sampler = WeightedRandomSampler(example_weights, len(train_set)*num, replacement=True)
    return sampler

def build_optimizer(model, optimizer, lr):
    if optimizer == "SGDP":
        optimizer = SGDP(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-5,
            momentum=0.9,
            nesterov=True)
    else:
        optimizer = AdamP(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )
    return optimizer

def build_scheduler(scheduler, optimizer):
    if scheduler == 'steplr':
        scheduler = StepLR(optimizer, 20, gamma=0.5)
    elif scheduler == 'cosinelr':
        scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=0.)
    return scheduler

def train(config=None):
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    with wandb.init(config=sweep_config):
        # -- sweep controller
        config = wandb.config

        # -- seed생성
        seed_everything(config.seed)
        
        # -- dir init
        save_dir = increment_path(os.path.join(config.model_dir, config.file))
        print("데이터 셋을 불러옵니다...")

        # -- dataset
        dataset_module = getattr(import_module("dataset"), config.dataset)  # default: BaseAugmentation
        dataset = dataset_module(
            data_dir=config.data_dir,
        )    
        # -- augmentation
        transform_module = getattr(import_module("dataset"), config.augmentation)  # default: BaseAugmentation
        transform = transform_module(
            resize=config.resize,
            mean=dataset.mean,
            std=dataset.std,
        )
        dataset.set_transform(transform)
        
        # -- initialize
        best_val_f1 = 0
        best_val_acc = 0
        best_val_loss = np.inf
        counter = 1
        patient = 2
        accumulation_steps = config.accumulation_steps
        oof_pred = None

        macro_f1 = F1Score("macro")

        # -- data_loader
        for i, (train_idx, valid_idx) in enumerate(dataset.k_split_dataset()):
            train_set = torch.utils.data.Subset(dataset, indices=train_idx)
            val_set   = torch.utils.data.Subset(dataset, indices=valid_idx)
            # weighted random sampling
            #sampler = weighted_sampler(train_set, config.num_sample)
            train_loader = DataLoader(
                train_set,
                batch_size=config.batch_size,
                num_workers=4,
                #sampler=sampler,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True
            )
            val_loader = DataLoader(
                val_set,
                batch_size=config.valid_batch_size,
                num_workers=2,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=True,
            )
            # -- model
            print("모델을 불러옵니다.")
            model_module = getattr(import_module("model"),"MyModel")
            model = model_module(
                model_name = config.model_name
            ).to(device)
            model = torch.nn.DataParallel(model)

            print("loss를 정의합니다.")
            # -- loss & metric        
            weights = [0.172, 0.3, 0.52, 0.129, 0.141, 0.367, 0.860, 1.504, 2.638, 
                       0.645, 0.705, 1.837, 0.860, 1.504, 2.638, 0.645, 0.705, 1.837]
            class_weights = torch.FloatTensor(weights).cuda()
            criterion = create_criterion(config.criterion, weight=class_weights)  # default: cross_entropy

            # -- optimizer
            optimizer = build_optimizer(model, config.optimizer, config.learning_rate)

            # -- scheduler
            scheduler = build_scheduler(config.scheduler, optimizer)

            # -- create dir
            try:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            except OSError:
                print ('Error: Creating directory. ' +  directory)

            for epoch in range(config.epochs):
                # train loop
                model.train()
                loss_value = 0
                matches = 0
                for idx, train_batch in enumerate(train_loader):
                    inputs, labels = train_batch
                    inputs, labels = inputs.to(device), labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                    loss.backward()
                    # -- Gradient Accumulation
                    if (idx+1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()             

                    loss_value += loss.item()
                    matches += (preds == labels).sum().item()
                    if (idx + 1) % config.log_interval == 0:
                        train_loss = loss_value / config.log_interval
                        train_acc = matches / config.batch_size / config.log_interval
                        current_lr = get_lr(optimizer)
                        f1 = macro_f1(preds, labels).item()
                        print(
                            f"Epoch[{epoch}/{config.epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"f1-score {f1:4.2} || training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr:4.5} || "
                        )
                        wandb.log({"train_loss":train_loss, "train_acc":train_acc, "f1-score":f1})
                        loss_value = 0
                        matches = 0

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    val_loss_items = []
                    val_acc_items = []
                    figure = None
                    for val_batch in val_loader:
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()
                        acc_item = (labels == preds).sum().item()
                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(val_set)
                    val_f1 = macro_f1(preds, labels).item()
                    best_val_loss = min(best_val_loss, val_loss)
                    if val_acc > best_val_acc:
                        print(f"New best model for f1 score : {val_acc:4.2%}! saving the best model..")
                        best_val_acc = val_acc
                        best_val_f1 = val_f1
                        torch.save(model.module.state_dict(), f"{save_dir}/best_{best_val_acc}.pth")
                    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                    print(
                        f"[Val] f1-score: {val_f1:4.2}, acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                        f"best f1: {best_val_f1:4.2}, best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} "
                    )
                    wandb.log({"val_loss":val_loss, "val_acc":val_acc, "val_f1":val_f1})
                    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
                        json.dump({"best acc":best_val_acc, "best loss":best_val_loss, "best f1":best_val_f1}, f, ensure_ascii=False, indent=4)
                    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model_name', type=str, default='vgg11', help='pretrained model name (default: vgg11)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: label_smoothing)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--file', default='exp', help='model save at {SM_MODEL_DIR}/{file}')
    
    args = parser.parse_args()
    sweep_config = create_sweep_config(args)
    sweep_id = wandb.sweep(sweep_config, project='efficientnet-sweeps2')
    wandb.agent(sweep_id, train, count=1)