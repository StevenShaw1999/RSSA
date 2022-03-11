from json.tool import main
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model import HED_Network

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

from torchvision.models.inception import inception_v3
from PIL import Image
import numpy as np
from scipy.stats import entropy
import argparse

def inception_score(img_dir, batch_size=100, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    dat = []
    N = 1000
    for i in range(N):
        img = np.array(Image.open(img_dir +'/img'+str(i) +'.png'))
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = 2 * img / 255.0 - 1
        dat.append(img)
    
    dat = torch.cat(dat, dim=0)

    assert batch_size > 0
    assert N >= batch_size
    
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').cuda()
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i in range(int(N / batch_size)):
        batch = dat[i* int(N / 10): (i+1) * int(N / 10), :, :, :].cuda()
        
        #batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

##########################################################

def HED_estimate(tenInput, HED_net):

	#assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	with torch.no_grad():
		return HED_net(tenInput.cuda()).cpu()
# end

##########################################################

def read_Image(img_name):
	return torch.from_numpy(numpy.ascontiguousarray(numpy.array(PIL.Image.open(img_name))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

def SCS_eval(args, HED_net):
    img_dir = args.img_pth 
    imgs_source = os.listdir(os.path.join(img_dir, 'source'))
    imgs_target = os.listdir(os.path.join(img_dir, 'target'))
    imgs_source_HED = os.path.join(img_dir, 'source_HED')
    imgs_target_HED = os.path.join(img_dir, 'target_HED')
    if not os.path.exists(imgs_source_HED):
        os.makedirs(imgs_source_HED)
    if not os.path.exists(imgs_target_HED):
        os.makedirs(imgs_target_HED)
    img_list = []
    for i in range(500):  
        image_name = os.path.join(os.path.join(img_dir, 'source'), imgs_source[i])
        img_list.append(read_Image(image_name).unsqueeze(0))
	#exit()
	#enInput = torch.FloatTensor()
    img_list = torch.cat(img_list, dim=0)
    tenOutput_list = []
    for i in range(20):
        tenOutput = HED_estimate(img_list[i*25:(i+1)*25], HED_net)
        tenOutput_list.append(tenOutput)
        tenOutput = torch.cat(tenOutput_list)
    for num, item in enumerate(tenOutput):
        PIL.Image.fromarray((item.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(f'%s/%s' % (imgs_source_HED, imgs_source[num]))

    img_list = []
    for i in range(500):  
        image_name = os.path.join(os.path.join(img_dir, 'target'), imgs_target[i])
        img_list.append(read_Image(image_name).unsqueeze(0))
    tenOutput_list = []
    img_list = torch.cat(img_list, dim=0)
    for i in range(20):
        tenOutput = HED_estimate(img_list[i*25:(i+1)*25], HED_net)
        tenOutput_list.append(tenOutput)
        tenOutput = torch.cat(tenOutput_list)
    for num, item in enumerate(tenOutput):
        PIL.Image.fromarray((item.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(f'%s/%s' % (imgs_target_HED, imgs_target[num]))
    score = 0
    for i in range(500):    
    
        img_s = np.array(Image.open(f'%s/img{i}.png' % imgs_source_HED)) 
        img_t = np.array(Image.open(f'%s/img{i}.png' % imgs_target_HED))
        img_s = torch.from_numpy(img_s).cuda().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) / 255
        img_t = torch.from_numpy(img_t).cuda().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) / 255

        sim = 2 * (img_s * img_t).sum() / (img_s**2 + img_t**2).sum()
        score += sim
    
    print('SCS Score: %.3f' % (score / 500))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_pth', type=str, default=None)
    parser.add_argument('--mode', type=str, default='IS')

    args = parser.parse_args()
    if args.mode == 'IS':
        IS_mean, IS_std = inception_score(args.img_pth, batch_size=100, resize=False, splits=5)
        print('Inception score: %.2f' % IS_mean)
    
    if args.mode == 'SCS':
        HED_net = HED_Network().cuda().eval()
        SCS_eval(args, HED_net)
        exit()