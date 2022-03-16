import argparse
import random

from matplotlib.image import imsave
import torch
import torch.nn as nn
from torchvision import utils
from model import Generator, Projection_module, Projection_module_church
from tqdm import tqdm
import sys
import os

def generate_gif(args, g_source, g_target, Proj_module):
    if args.load_noise:
        noise = torch.load(args.load_noise).cuda()
        
    else:
        noise = torch.randn(args.n_sample, args.latent).cuda()

    with torch.no_grad():

        n_steps = args.n_steps
        step = float(1)/n_steps
        n_paths = noise.size(0)
        for t in range(n_paths):
            print(t)
            if t != (n_paths - 1):
                z1, z2 = torch.unsqueeze(
                    noise[t], 0), torch.unsqueeze(noise[t+1], 0)
            else:
                z1, z2 = torch.unsqueeze(
                    noise[t], 0), torch.unsqueeze(noise[0], 0)

            for i in range(n_steps):
                alpha = step*i
                z = z2*alpha + (1-alpha)*z1
                sample_s, _ = g_source([z], randomize_noise=False)
                w = [g_target.module.style(z)]
                w = [Proj_module.modulate(item) for item in w]
                sample_t, _= g_target(w, input_is_latent=True, randomize_noise=False)

                utils.save_image(
                    sample_s,
                    f'%s/sample%d.png' % (args.save_source, (t*n_steps) + i) ,
                    normalize=True,
                    range=(-1, 1),
                )

                utils.save_image(
                    sample_t,
                    f'%s/sample%d.png' % (args.save_target, (t*n_steps) + i),
                    normalize=True,
                    range=(-1, 1),
                )


def generate_imgs(args, g_source, g_target, Proj_module):

    with torch.no_grad():
        
        if args.load_noise:
            sample_z = torch.load(args.load_noise)
        else:
            sample_z = torch.randn(args.n_sample, args.latent).cuda()

        sample_s, _ = g_source([sample_z], input_is_latent=False, randomize_noise=False)
        w = [g_target.module.style(sample_z)]
        w = [Proj_module.modulate(item) for item in w]
        sample_t, _= g_target(w, input_is_latent=True, randomize_noise=False)

        utils.save_image(
            sample_s,
            f'%s/sample_s.png' % args.save_source,
            nrow=5,
            normalize=True,
            range=(-1, 1),
        )

        utils.save_image(
            sample_t,
            f'%s/sample_t.png' % args.save_target,
            nrow=5,
            normalize=True,
            range=(-1, 1),
        )


def generate_img_pairs(args, g_source, g_target, Proj_module):
    
    with torch.no_grad():
        sample_z = torch.randn(args.SCS_samples, args.latent).cuda()
        for i in range(10):
            print(i)
            w = [g_target.module.style(sample_z[i* int(args.SCS_samples / 10): (i+1)*int(args.SCS_samples / 10)])]
            w = [Proj_module.modulate(item) for item in w]
            sample_t, _= g_target(w, input_is_latent=True, randomize_noise=False)
            sample_s, _ = g_source([sample_z[i* int(args.SCS_samples / 10): (i+1)*int(args.SCS_samples / 10)]], input_is_latent=False, randomize_noise=False)
            
            for (num, (img_s, img_t)) in enumerate(zip(sample_s, sample_t)):
                utils.save_image(
                    img_s,
                    f'%s/img%d.png' % (args.save_source, (i* int(args.SCS_samples / 10)) + num) ,
                    normalize=True,
                    range=(-1, 1),
                )

                utils.save_image(
                    img_t,
                    f'%s/img%d.png' % (args.save_target, (i * int(args.SCS_samples / 10)) + num) ,
                    normalize=True,
                    range=(-1, 1),
                )


def generate_imgs_4IS(args, g_target, Proj_module):
    
    with torch.no_grad():
        
        sample_z = torch.randn(args.IS_sample, args.latent).cuda()
        step = int(args.IS_sample / 50)
        batch = 50
        for i in range(int(args.IS_sample / 50)):
            print(i)
            w = [g_target.module.style(sample_z[i*batch: (i+1)*batch])]
            w = [Proj_module.modulate(item) for item in w]
            sample_t, _= g_target(w, input_is_latent=True, randomize_noise=False)

            for (num, img) in enumerate(sample_t):
                utils.save_image(
                    img,
                    f'%s/img%d.png' % (args.save_target, (i * batch) + num) ,
                    normalize=True,
                    range=(-1, 1),
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--SCS_samples', type=int, default=500, help='number of image pairs to eval SCS')
    parser.add_argument('--n_sample', type=int, default=25, help='number of fake images to be sampled')
    parser.add_argument('--IS_sample', type=int, default=10000, help='number of fake images to be sampled for IS')
    parser.add_argument('--n_steps', type=int, default=40, help="determines the granualarity of interpolation")
    parser.add_argument('--ckpt_source', type=str, default=None)
    parser.add_argument('--ckpt_target', type=str, default=None)
    parser.add_argument('--mode', type=str, default='viz_imgs', help='viz_imgs,viz_gif,eval_IS,eval_SCS')
    parser.add_argument('--load_noise', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--target', type=str, default='VanGogh', help='target domain')
    parser.add_argument('--task', type=int, default=10)
    parser.add_argument('--source', type=str, default='face', help='source domain')
    parser.add_argument('--latent_dir', type=str)

    torch.manual_seed(10)
    random.seed(10)
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    args.exp_name = args.target
    if args.source == 'church':
        Proj_module = Projection_module_church(args)
    if args.source == 'face':
        Proj_module = Projection_module(args)


    if args.mode == 'viz_imgs' or args.mode == 'eval_SCS':
        temp_str = f"%s2%s_%s" % (args.source, args.target, str(args.task))
        imsave_path_source = os.path.join(args.mode, temp_str, 'source')
        imsave_path_target = os.path.join(args.mode, temp_str, 'target')
        if not os.path.exists(imsave_path_source):
            os.makedirs(imsave_path_source)
        if not os.path.exists(imsave_path_target):
            os.makedirs(imsave_path_target)
        
        args.save_source = imsave_path_source
        args.save_target = imsave_path_target
    
    if args.mode == 'viz_gif':
        temp_str = f"%s_%s" % (args.source, str(args.task))
        imsave_path_source = os.path.join(args.mode, temp_str, 'source')
        imsave_path_target = os.path.join(args.mode, temp_str, args.target)
        if not os.path.exists(imsave_path_source):
            os.makedirs(imsave_path_source)
        if not os.path.exists(imsave_path_target):
            os.makedirs(imsave_path_target)
        args.save_source = imsave_path_source
        args.save_target = imsave_path_target

    if args.mode == 'eval_IS':
        temp_str = f"%s2%s_%s" % (args.source, args.target, str(args.task))
        imsave_path_target = os.path.join(args.mode, temp_str)
        if not os.path.exists(imsave_path_target):
            os.makedirs(imsave_path_target)
        
        args.save_target = imsave_path_target
    
    
    # loading source model if available
    if args.ckpt_source is not None:
        g_source = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        checkpoint = torch.load(args.ckpt_source)
        g_source.load_state_dict(checkpoint['g_ema'], strict=False)

    # loading target model if available
    if args.ckpt_target is not None:
        g_target = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        g_target = nn.parallel.DataParallel(g_target)
        checkpoint = torch.load(args.ckpt_target)
        g_target.load_state_dict(checkpoint['g_ema'], strict=False)
    
    if args.mode == 'viz_imgs':

        generate_imgs(args, g_source, g_target, Proj_module)
    
    if args.mode == 'eval_IS':
    
        generate_imgs_4IS(args, g_target, Proj_module)
    

    if args.mode == 'eval_SCS':
    
        generate_img_pairs(args, g_source, g_target, Proj_module)



    elif args.mode == 'viz_gif':
        generate_gif(args, g_source, g_target, Proj_module)
