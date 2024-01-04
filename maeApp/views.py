import json

from django.http import HttpResponse

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from django.forms.models import model_to_dict
from django.db.models import Q
from datetime import datetime
from django.contrib.auth.decorators import login_required
import sys
import os, time
import requests
import cv2
import torch
import numpy as np
from mae import models_mae
import matplotlib.pyplot as plt
from PIL import Image
# define the utils
from .forms import ImgUploadModelForm, ModelPthUploadModelForm
from mae2.main_mae import run_train

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    return im_paste[0]
    # # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]
    #
    # plt.subplot(1, 4, 1)
    # show_image(x[0], "original")
    #
    # plt.subplot(1, 4, 2)
    # show_image(im_masked[0], "masked")
    #
    # plt.subplot(1, 4, 3)
    # show_image(y[0], "reconstruction")
    #
    # plt.subplot(1, 4, 4)
    # show_image(im_paste[0], "reconstruction + visible")
    #
    # plt.show()


# 图片读入处理函数 tiff改jpg
def loadimage(img_url):
    img = Image.open(img_url)
    img.convert('RGB')
    img = img.resize((224, 224))
    img.save('media/files/input_image.jpg')
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    return img


@login_required
def mae(request):
    in_image_url = 'static/img/loading.png'
    out_image_url = 'static/img/loading.png'
    # 现有模型 时间
    pthname = None
    ctime = None
    mtime = None
    if len(os.listdir('media/files/pth')) != 0:
        pthname = os.listdir('media/files/pth')[0]
        ctime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat('media/files/pth/' + pthname).st_ctime))
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat('media/files/pth/' + pthname).st_mtime))

    data_json=[]
    if request.method == "GET":
        a = request.GET.get("train", '')
        if a != '':
            run_train()
            file = open('log/data.json','r')
            for line in file.readlines():
                dic = json.loads(line)
                data_json.append(dic['loss'])
            file.close()
    if request.method == "POST":
        imgform = ImgUploadModelForm(request.POST, request.FILES)
        mpform = ModelPthUploadModelForm(request.POST, request.FILES)
        if mpform.is_valid():
            mpform.save()  # 一句话足以
            pthname = os.listdir('media/files/pth')[0]
            ctime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat('media/files/pth/' + pthname).st_ctime))
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat('media/files/pth/' + pthname).st_mtime))
        if imgform.is_valid():
            imgfile = imgform.save()
            # in_image_url
            # TODO 预测
            model_mae_gan = prepare_model('media/files/pth/'+pthname, 'mae_vit_large_patch16')
            img = loadimage(imgfile.file)

            in_image_url = 'media/files/input_image.jpg'
            torch.manual_seed(2)
            tensorimg = run_one_image(img, model_mae_gan)
            npimg = tensorimg.numpy()
            outimage = Image.fromarray(np.uint8(npimg))
            outimage.save('media/files/output_image.jpg')
            out_image_url = 'media/files/output_image.jpg'
    else:
        imgform = ImgUploadModelForm()
        mpform = ModelPthUploadModelForm()


    return render(request, 'mae.html',
                  {'ImgForm': imgform, 'ModelPthForm': mpform, 'heading': 'Upload files with ModelForm',
                   'ctx': 'false', 'in_image_url': in_image_url, 'out_image_url': out_image_url,
                   'pthname': pthname,  'ctime': ctime,  'mtime': mtime, 'data_json':data_json}
                  )
