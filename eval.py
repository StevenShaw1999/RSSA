from json.tool import main
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_pth', type=str, default=None)
    parser.add_argument('--mode', type=str, default='IS')

    args = parser.parse_args()
    if args.mode == 'IS':
        IS_mean, IS_std = inception_score(args.img_pth, batch_size=100, resize=False, splits=5)
        print('Inception score: %.2f' % IS_mean)
    

