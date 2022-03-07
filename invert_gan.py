import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from lpips import LPIPS

from model import Generator


parser = argparse.ArgumentParser('Invertor')
# training params
parser.add_argument('--expname', type=str, default='exp1', help='experiment name')
parser.add_argument('--expdir', type=str, default='exps', help='dirs of experiments')
#parser.add_argument('--imagename', type=str, required=True, help='input image name')
parser.add_argument('--stylegan2_path', type=str, default='data/stylegan2-ffhq-config-f.pt', help='path of pretrianed stylegan model')
parser.add_argument('--iter_num', type=int, default=4002, help='iteration steps')
parser.add_argument('--learning_rate', type=float, default=1e-1, help='learning rate')
parser.add_argument('--image_dir', type=str, default='data/', help='path to inverted images')
parser.add_argument('--latent_dir', type=str, default='latent/', help='path to inverted images')
args = parser.parse_args()

def preprocess(images, channel_order='RGB'):
    """Preprocesses the input images if needed.
    This function assumes the input numpy array is with shape [batch_size,
    height, width, channel]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, channel, height, width].
    NOTE: The channel order of input images is always assumed as `RGB`.
    Args:
      images: The raw inputs with dtype `numpy.uint8` and range [0, 255].
    Returns:
      The preprocessed images with dtype `numpy.float32` and range
        [-1, 1].
    """
    # input : numpy, np.uint8, 0~255, RGB, BHWC
    # output : numpy, np.float32, -1~1, RGB, BCHW

    image_channels = 3
    max_val = 1.0
    min_val = -1.0

    if image_channels == 3 and channel_order == 'BGR':
      images = images[:, :, :, ::-1]
    images = images / 255.0 * (max_val - min_val) + min_val
    images = images.astype(np.float32).transpose(0, 3, 1, 2)
    return images


class Invertor():
    def __init__(self,options):
        self.options = options
        self.Up = nn.Upsample(256)
        self.device = torch.device('cuda')
        self.exppath = os.path.join(self.options.expdir, self.options.expname)
        os.makedirs(self.exppath, exist_ok=True)

        # load stylegan2
        self.G = Generator(256, 512, 8, channel_multiplier=2
        ).to(self.device)

        ckpt_source = torch.load(args.stylegan2_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(ckpt_source["g_ema"], strict=False)

        
        with torch.no_grad():
            self.find_avg_latent()
        self.avg_latent = self.avg_latent.view(1, 1, -1).clone()
        self.avg_latent = self.avg_latent.repeat(1, self.G.n_latent, 1)

        # setup image transform
        self.image_transforms = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
        
        # setup criterion
        self.lpips_criterion = LPIPS(net='vgg').to(self.device).eval()
        self.MSE_criterion = nn.MSELoss().to(self.device)

    def find_avg_latent(self,):
        self.avg_latent = torch.randn((5120,512)).to(self.device)
        with torch.no_grad():
            self.avg_latent = torch.mean(self.G.style(self.avg_latent), dim=0, keepdim=True)



    def write_summaries(self, results, step):
        for k in results.items():
            if 'loss' in k:
                self.logger.add_scalar(f'{k}', results[k], step)
            elif 'image' in k:
                self.logger.add_images(f'{k}', results[k], step)
        return


    def read_image(self, imagename):
        image = cv2.imread(imagename)
        image_target = torch.from_numpy(preprocess(image[np.newaxis, :], channel_order='BGR')).cuda()
        image_target = self.Up(image_target)
        return image_target
    

    def tensor2numpy(self, images):
        """ we assume the shape of image is (1, C, H, W), and it's a cuda pytorch tensor
        """
        images = torch.clamp(images.detach(), min=-1, max=1)
        images = ((images+1)/2)*255
        images = images.permute(0,2,3,1).detach().cpu().numpy().astype('uint8')
        return images


    def initial_latentcode(self, latent_type):
        if latent_type == 'randn':
            return torch.randn((1,18,512)).to(self.device)
        elif latent_type == 'zero':
            return torch.zeros((1,18,512)).to(self.device)
        elif latent_type == 'mean':
            return torch.from_numpy(np.load('data/mean_latent.npy')).float().to(self.device).unsqueeze(0)
        elif latent_type == 'mean_ckpt':
            return self.mean_latent
        else:
            raise NotImplementedError


    def run(self,):
        image_list = os.listdir(args.image_dir)
        
        for i in range(5, 10):
            print(args.image_dir + image_list[i])
            image = self.read_image(args.image_dir + image_list[i])
            self.avg_latent_sub = self.avg_latent.clone()
            self.avg_latent_sub.requires_grad = True
            optimizer = torch.optim.Adam([self.avg_latent_sub], lr=self.options.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.5)
            for step in tqdm(range(self.options.iter_num)):
                decoded_image, _ = self.G([self.avg_latent_sub], input_is_latent=True, return_latents=True, randomize_noise=False)
                lpipsloss = self.lpips_criterion(decoded_image, image)
                mseloss = self.MSE_criterion(decoded_image, image)

                loss = lpipsloss + mseloss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                dataitems = {'lpipsloss': lpipsloss, 
                            'mseloss': mseloss}
                if step % 500 == 0 or step == self.options.iter_num:
                    self.write_summaries(dataitems, step)
                    #print(step)
                    decoded_image_np = self.tensor2numpy(decoded_image)
                    decoded_image_np = Image.fromarray(decoded_image_np[0])
                    decoded_image_np.save(args.latent_dir + f'images/{i}.png')
                    np.save(args.latent_dir + f'latent/{i}_latentcode.npy', self.avg_latent_sub.detach().cpu().numpy())
        print('Finished')
            

if __name__ == '__main__':
    options = parser.parse_args()
    invertor = Invertor(options)
    invertor.run()