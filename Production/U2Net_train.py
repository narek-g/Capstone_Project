import os
import sys
import time
import yaml
import cv2
import pprint
import traceback
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

from data.custom_dataset_data_loader import CustomDatasetDataLoader, sample_data

from utils.tensorboard_utils import board_add_images
from utils.saving_utils import save_checkpoints
from utils.saving_utils import load_checkpoint, load_checkpoint_mgpu

# from networks import U2NET
from model import U2NET

def options_printing_saving(opt):
    os.makedirs(opt.logs_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "checkpoints"), exist_ok=True)

        

def training_loop(opt):

    if opt.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK"))
        # Unique only on individual node.
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
        local_rank = 0

    
    model_name = 'u2net'
    u_net = U2NET(in_ch=3, out_ch=4)

    if torch.cuda.is_available():
        u_net.cuda()
        print('Cuda available..')


    # initialize optimizer
    optimizer = optim.Adam(
        u_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    custom_dataloader = CustomDatasetDataLoader()
    custom_dataloader.initialize(opt)
    loader = custom_dataloader.get_loader()

    if local_rank == 0:
        dataset_size = len(custom_dataloader)
        print("Total number of images avaliable for training: %d" % dataset_size)
        print("Entering training loop!")

    # loss function
    weights = np.array([1, 1.5, 1.5, 1.5], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)
    
    def loss_fusion(d0,d1,d2,d3,d4,d5,d6,label):
        loss0 = loss_CE(d0, label)
        loss1 = loss_CE(d1, label)
        loss2 = loss_CE(d2, label)
        loss3 = loss_CE(d3, label)
        loss4 = loss_CE(d4, label)
        loss5 = loss_CE(d5, label)
        loss6 = loss_CE(d6, label)
        del d1, d2, d3, d4, d5, d6
        total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return(total_loss, loss0)

    epochs = range(opt.iter)
    get_data = sample_data(loader)

    start_time = time.time()
    # Main training loop
    for itr in epochs:
        data_batch = next(get_data)
        image, label = data_batch
        # image = Variable(image.to(device))
        image = Variable(image)
        label = label.type(torch.long)
        # label = Variable(label.to(device))
        label = Variable(label)

        d0, d1, d2, d3, d4, d5, d6 = u_net(image)
        (total_loss, loss0) = loss_fusion(d0,d1,d2,d3,d4,d5,d6,label)

        for param in u_net.parameters():
            param.grad = None

        total_loss.backward()
        
        if opt.clip_grad != 0:
            nn.utils.clip_grad_norm_(u_net.parameters(), opt.clip_grad)
        optimizer.step()

        if local_rank == 0:
            # printing and saving work
            if itr % opt.print_freq == 0:
                pprint.pprint(
                    "[step-{:08d}] [time-{:.3f}] [total_loss-{:.6f}]  [loss0-{:.6f}]".format(
                        itr, time.time() - start_time, total_loss, loss0
                    )
                )

            if itr % opt.save_freq == 0:
                save_checkpoints(opt, itr, u_net)

    print("Training done!")



if __name__ == "__main__":

    opt.name = "training_cloth_segm_u2net_aws_ex3"  # Expriment name
    opt.image_folder = "/ubuntu/home/capstone/training_data/train"  # image folder path
    opt.df_path = "/ubuntu/home/capstone/training_data/train.csv"  # label csv path
    opt.isTrain = True

    opt.fine_width = 192 * 4
    opt.fine_height = 192 * 4

    # Mean std params
    opt.mean = 0.5
    opt.std = 0.5

    opt.batchSize = 2  # 12
    opt.nThreads = 1  # 3
    opt.max_dataset_size = float("inf")

    opt.serial_batches = False
    opt.continue_train = True
    if opt.continue_train:
        opt.unet_checkpoint = "prev_checkpoints/cloth_segm_unet_surgery.pth"

    opt.save_freq = 10
    opt.print_freq = 10
    opt.image_log_freq = 10

    opt.iter = 10000
    opt.lr = 0.0002
    opt.clip_grad = 5

    opt.logs_dir = osp.join("logs", self.name)
    opt.save_dir = osp.join("results", self.name)

    try:
        set_seed(400)
        training_loop(opt)
        print("Training complete..")

