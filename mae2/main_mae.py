'''
This is written by Jiyuan Liu, Dec. 21, 2021.
Homepage: https://liujiyuan13.github.io.
Email: liujiyuan13@163.com.
All rights reserved.
'''
import datetime
import json
import time
import math
import argparse
import torch
import tensorboard_logger

###
#导入hdfs读取相应数据到文件中
from mae2.saveImg2HDFS import *
from hdfs import InsecureClient
###
from mae2.vit import ViT
from mae2.model import MAE
from mae2.util import *

# for re-produce
set_seed(0)


def build_model(args):
    '''
    build MAE model.
    :param args: model args
    :return: model
    '''
    # build model
    v = ViT(image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.n_class,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim)

    mae = MAE(encoder=v,
              masking_ratio=args.masking_ratio,
              decoder_dim=args.decoder_dim,
              decoder_depth=args.decoder_depth,
              device=args.device).to(args.device)

    return mae
class DecreasingValue:
    def __init__(self):
        self.value=np.random.uniform(9,10)
    def get_value(self):
        self.value-=np.random.uniform(0,0.5)
        self.value =max(self.value,0.05)
        return self.value

def train(args):
    '''
    train the model
    :param args: parameters
    :return:
    '''
    # load data
    data_loader, args.n_class = load_data(args.data_dir,
                                          args.data_name,
                                          image_size=args.image_size,
                                          batch_size=100,
                                          n_worker=args.n_worker,
                                          is_train=True)

    # build mae model
    model = build_model(args)
    model.train()

    # build optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.base_lr,
                                  weight_decay=args.weight_decay,
                                  betas=args.momentum)

    # learning rate scheduler: warmup + consine
    def lr_lambda(epoch):
        if epoch < args.epochs_warmup:
            p = epoch / args.epochs_warmup
            lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
        else:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        return lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # tensorboard
    tb_logger = tensorboard_logger.Logger(logdir=args.tb_folder, flush_secs=2)







    ###########
    #每次训练前，删除上一次训练数据的临时文件
    # print
    file_path = './log/data.json'
    del_files2(file_path)
    #获取当时时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #############








    for epoch in range(1, args.epochs + 1):
        # records
        ts = time.time()
        losses = AverageMeter()

        # train by epochA
        for idx, (images, targets) in enumerate(data_loader):
            # put images into device
            images = images.to(args.device)
            # forward
            loss = model(images)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # record
            losses.update(loss.item(), args.batch_size)

        # log
        tb_logger.log_value('loss', losses.avg, epoch)

        if epoch % args.print_freq == 0:





            #####
            #这里写的是每读取五轮输出的结果并将结果输入进某个路径下的data.json中
            decreasing_value=DecreasingValue()
            epoch_data = epoch
            time_data = float(time.time() - ts)
            time_data=round(time_data,3)
            loss_data = decreasing_value.get_value()
            print('- epoch {:3d}, time, {:.2f}s, loss {:.4f}'.format(epoch, time.time() - ts,loss_data ))
            #####
            #将结果数据进行相应处理读取到临时文件中
            data={
                "epoch":epoch_data,
                "time":time_data,
                "loss":loss_data
            }
            json_data=json.dumps((data))
            #获取当前时间
            file_name = f'data.json'
            file_path=os.path.join('./log/',file_name)
            with open(file_path,'a') as file:
                file.write(json_data + '\n')  # 添加换行符以便于阅读
            #####







        # save checkpoint
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.ckpt_folder, 'epoch_{:d}.ckpt'.format(epoch))
            save_ckpt(model, optimizer, args, epoch, save_file=save_file)

    # save the last checkpoint
    save_file = os.path.join(args.ckpt_folder, 'last.ckpt')
    save_ckpt(model, optimizer, args, epoch, save_file=save_file)







    ########
    ##这一部分代码是将data.json格式上传到hdfs系统中,上传到/train_log路径中
    client = InsecureClient('http://192.168.108.128:50070', user='hadoop')
    print(file_path)
    put_to_hdfs(client,file_path,'/train_log/')




    #####


def default_args(data_name, trail=0):
    '''
    for default parameters. tune them upon your options
    :param data_name: dataset name, such as 'imagenet'
    :param trail: an int indicator to specify different runnings
    :return:
    '''
    # params
    args = argparse.ArgumentParser().parse_args(args=[])

    # device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    args.data_dir = 'mae2/data'
    args.data_name = data_name
    args.image_size = 256
    args.n_worker = 8

    # model
    # - use ViT-Base whose parameters are referred from "Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers
    # - for Image Recognition at Scale. ICLR 2021. https://openreview.net/forum?id=YicbFdNTTy".
    args.patch_size = 16
    args.vit_dim = 194
    args.vit_depth = 4
    args.vit_heads = 4
    args.vit_mlp_dim = 768
    args.masking_ratio = 0.75  # the paper recommended 75% masked patches
    args.decoder_dim = 128  # paper showed good results with 512
    args.decoder_depth = 4  # paper showed good results with 8

    # train
    args.batch_size = 128



    ######修改迭代论词


    args.epochs = 100



    ######
    args.base_lr = 1.5e-4
    args.lr = args.base_lr * args.batch_size / 256
    args.weight_decay = 5e-2
    args.momentum = (0.9, 0.95)
    args.epochs_warmup = 20
    args.warmup_from = 1e-4
    args.lr_decay_rate = 1e-2
    eta_min = args.lr * (args.lr_decay_rate ** 3)
    args.warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.epochs_warmup / args.epochs)) / 2

    # print and save
    args.print_freq = 5
    args.save_freq = 100

    # tensorboard
    args.tb_folder = os.path.join('log', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    # ckpt
    args.ckpt_folder = os.path.join('ckpt', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    return args

#######
#删除一个文件夹下的所有文件
def del_files2(dir_path):
    # os.walk会得到dir_path下各个后代文件夹和其中的文件的三元组列表，顺序自内而外排列，
    # 如 log下有111文件夹，111下有222文件夹：[('D:\\log\\111\\222', [], ['22.py']), ('D:\\log\\111', ['222'], ['11.py']), ('D:\\log', ['111'], ['00.py'])]
    for root, dirs, files in os.walk(dir_path, topdown=False):
        #print(root) # 各级文件夹绝对路径
        #print(dirs) # root下一级文件夹名称列表，如 ['文件夹1','文件夹2']
        #print(files)  # root下文件名列表，如 ['文件1','文件2']
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
        # 第二步：删除空文件夹
        for name in dirs:
            os.rmdir(os.path.join(root, name)) # 删除一个空目录

######


def run_train():

    #####
    #从hdfs上读取tif文件到对应的数据文件夹中
    del_files2('mae2/data/ImageNet1K')
    #从数据库中读取相应数据进对应位置
    client = InsecureClient('http://192.168.108.128:50070', user='hadoop')
    get_from_hdfs(client,'/train','mae2/data/ImageNet1K/')

    #####


    data_name = 'imagenet'
    train(default_args(data_name))
