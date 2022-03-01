import argparse
import math
import random
import os
import sys
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.nn.modules import loss
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import viz
from copy import deepcopy
import numpy



try:
    import wandb

except ImportError:
    wandb = None


from model import Generator, Extra, Projection_module, Projection_module_church, avg_conv
from model import Patch_Discriminator as Discriminator  # , Projection_head
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def get_subspace(args, init_z, vis_flag=False):
    std = args.subspace_std
    bs = args.batch if not vis_flag else args.n_sample
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    return z


def train(args, loader, generator, discriminator, extra, g_optim, d_optim, e_optim, g_ema, device, g_source, d_source):
    loader = sample_data(loader)

    imsave_path = os.path.join('samples', args.exp)
    model_path = os.path.join('checkpoints', args.exp)

    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if args.proj != 'None':
        Proj_module = Projection_module_church(args)

    # this defines the anchor points, and when sampling noise close to these, we impose image-level adversarial loss (Eq. 4 in the paper)
    init_z = torch.randn(args.n_train, args.latent, device=device)
    pbar = range(args.iter)
    sfm = nn.Softmax(dim=1)
    kl_loss = nn.KLDivLoss()
    sim = nn.CosineSimilarity()
    pool_dict = {}
    pool_dict[32] = nn.AdaptiveAvgPool2d((16, 16))
    pool_dict[64] = nn.AdaptiveAvgPool2d((32, 32))
    pool_dict[128] = nn.AdaptiveAvgPool2d((64, 64))
    pool_dict[256] = nn.AdaptiveAvgPool2d((128, 128))
    
    pool_dict_sp = {}
    pool_dict_sp[32] = nn.AdaptiveAvgPool2d((16, 16))
    pool_dict_sp[64] = nn.AdaptiveAvgPool2d((16, 16))
    pool_dict_sp[128] = nn.AdaptiveAvgPool2d((64, 64))
    pool_dict_sp[256] = nn.AdaptiveAvgPool2d((64, 64))

    Avg = avg_conv().cuda()
    Loss_fn = nn.SmoothL1Loss().cuda()
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}


    g_module = generator.module
    d_module = discriminator
    g_ema_module = g_ema.module

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0


    # this defines which level feature of the discriminator is used to implement the patch-level adversarial loss: could be anything between [0, args.highp] 
    lowp, highp = 0, args.highp

    # the following defines the constant noise used for generating images at different stages of training
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    requires_grad(g_source, False)
    requires_grad(d_source, False)
    sub_region_z = get_subspace(args, init_z.clone(), vis_flag=True)
    for idx in pbar:
        i = idx + args.start_iter
        which = i % args.subspace_freq # defines whether we sample from anchor region in this iteration or other
        #if args.proj != 'None':
        #    Proj_module.adjust_sub(total_iter=args.iter, n_iter=i)
        #    continue
        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(extra, True)

        if which > 0:
            # sample normally, apply patch-level adversarial loss
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            # sample from anchors, apply image-level adversarial loss
            noise = [get_subspace(args, init_z.clone())]

        if args.proj != 'None':
            w = [generator.module.style(item) for item in noise]
            w = [Proj_module.modulate(item) for item in w]
            fake_img, _ = generator(w, input_is_latent=True)
        else: 
            fake_img, _ = generator(noise)

        if args.augment:
            real_img, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        real_pred, _ = discriminator(
            real_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp), real=True)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        extra.zero_grad()
        d_loss.backward()
        d_optim.step()
        e_optim.step()

        if args.augment and args.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred, _ = discriminator(
                real_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            extra.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()
            e_optim.step()
        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(extra, False)
        if which > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            noise = [get_subspace(args, init_z.clone())]

        if args.proj != 'None':
            w = [generator.module.style(item) for item in noise]
            w = [Proj_module.modulate(item) for item in w]
            fake_img, _ = generator(w, input_is_latent=True)

        else:  fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        g_loss = g_nonsaturating_loss(fake_pred)

        # distance consistency loss
        if args.self_sim_loss_new != 'None':
            with torch.set_grad_enabled(False):
                z = torch.randn(args.feat_const_batch, args.latent, device=device)
                if args.proj != 'None':
                    w = [g_source.module.style(z)]
            #w = [generator.module.style(item) for item in noise]
                    w = [Proj_module.modulate(item) for item in w]
                    #print('Here')
                    #exit()
                    _, feat_source = g_source(w, input_is_latent=True, return_feats=True, randomize_noise=False)
                else:
                    _, feat_source = g_source([z], return_feats=True, randomize_noise=False)
            if args.proj != 'None':
                w = [generator.module.style(z)]
            #w = [generator.module.style(item) for item in noise]
                w = [Proj_module.modulate(item) for item in w]
                _, feat_target = generator(w, input_is_latent=True, return_feats=True, randomize_noise=False)
            else:
                _, feat_target = generator([z], return_feats=True, randomize_noise=False)
            loss_d = 0
            #Corr_mat = []
            for idxx, (feat_s, feat_t) in enumerate(zip(feat_source, feat_target)):
                feat_size = feat_s.size(2)
                vector_len = feat_s.size(1)
    
                feat = torch.cat([feat_s, feat_t], dim=0)
                batchsize = feat.size(0)
                feat = F.normalize(feat, dim=1)
                if feat_size <= 32:
                    #feat = feat
                    feat = feat.reshape(feat.size(0), feat.size(1), feat_size ** 2)
                    feat = feat.permute(0, 2, 1)
                    self_sim = torch.matmul(feat, feat.transpose(1,2))
                    self_sim = torch.chunk(self_sim, 2, dim=0)
                    loss_d += Loss_fn(self_sim[0], self_sim[1])

                elif feat_size >= 64 and feat_size <= 256:
                    #feat = F.normalize(feat, dim=1)
                    feat = pool_dict[feat_size](feat)
                    feat_size = int(feat_size / 2)
                    window_size = int(feat_size / 2)
                    strid = int(feat_size / 2)
                    ud = int(window_size / 2)
                    unfold_feat = F.unfold(feat, (window_size, window_size), stride=strid)
                    patch_num = unfold_feat.size(-1)
                    unfold_feat = unfold_feat.resize(batchsize, vector_len, window_size * window_size, patch_num)
                    #unfold_feat = F.normalize(unfold_feat, dim=1)
                    unfold_feat = unfold_feat.permute(0,3,2,1).reshape(batchsize*patch_num, window_size * window_size, vector_len)
                    self_sim = torch.matmul(unfold_feat, unfold_feat.transpose(1,2)).reshape(batchsize, patch_num, window_size * window_size, window_size * window_size)
                    #print(self_sim.size())
                    self_sim = torch.chunk(self_sim, 2, dim=0)
                    
                    #topk_ind = torch.topk(slef_sim[1], int(sim_t.size(1) / 4) ,dim=1)[1]
                    loss_d += Loss_fn(self_sim[0], self_sim[1]) 
                    #(1 - idxx / 30) *
                    feat = feat[:, :, ud:feat_size-ud, ud:feat_size-ud]
                    #print(feat.size())
                    unfold_feat = F.unfold(feat, (window_size, window_size), stride=strid)
                    patch_num = unfold_feat.size(-1)
                    unfold_feat = unfold_feat.resize(batchsize, vector_len, window_size * window_size, patch_num)
                    #unfold_feat = F.normalize(unfold_feat, dim=1)
                    unfold_feat = unfold_feat.permute(0,3,2,1).reshape(batchsize*patch_num, window_size * window_size, vector_len)
                #print(torch.sum(item * item, dim=2))
                #print(item.size())
                    self_sim = torch.matmul(unfold_feat, unfold_feat.transpose(1,2)).reshape(batchsize, patch_num, window_size * window_size, window_size * window_size)
                    #print(self_sim.size())
                    self_sim = torch.chunk(self_sim, 2, dim=0)
                    loss_d += Loss_fn(self_sim[0], self_sim[1]) 
            
                else:
                    continue
        #exit()
        if args.self_sim_loss_new != 'None':
            g_loss = g_loss + 0.8 * loss_d
        


        if args.sp_inter_sim != 'None' and (i-1) % 4 == 0:
            loss_sp_inter_sim = 0
            #z_ori[2] = torch.sum(z_ori[:3]) / 3
            with torch.no_grad():
                z = torch.randn(1, args.latent, device=device).repeat(args.feat_const_batch, 1)
                #z_n = torch.randn(args.feat_const_batch, args.latent, device=device) / 9
                z_n = torch.randn(args.feat_const_batch, args.latent, device=device) 
                z = z * 0.85 + z_n * 0.15
                if args.proj != 'None':
                    w = [g_source.module.style(z)]
                    w = [Proj_module.modulate(item) for item in w]
                    _, feat_source = g_source(w, input_is_latent=True, return_feats=True, randomize_noise=False)
                else:
                    #_, feat_source = g_source([z], return_feats=True, randomize_noise=False)
                    _, feat_source = g_source([z_n], return_feats=True, randomize_noise=False)
                
            if args.proj != 'None':
                w = [generator.module.style(z)]
            #w = [generator.module.style(item) for item in noise]
                w = [Proj_module.modulate(item) for item in w]
                _, feat_target = generator(w, input_is_latent=True, return_feats=True, randomize_noise=False)
            else:
                _, feat_target = generator([z], return_feats=True, randomize_noise=False)
            for idxx, (feat_s, feat_t) in enumerate(zip(feat_source, feat_target)):
                #print(torch.sum(torch.abs(feat_s_cons - feat_t_cons)))
                #print(torch.sum(torch.abs(feat_s - feat_t)))
                feat_size_ori = feat_s.size(2)
                if feat_size_ori > 128:
                    continue
                
                feat_s = F.normalize(feat_s, dim=1)
                feat_t = F.normalize(feat_t, dim=1)
                #batchsize, vector_len, _, _ = feat_s.size()
                if feat_size_ori == 64:
                    #print(feat_s.size())
                    unfold_feat_s = F.unfold(feat_s, (16, 16), stride=16)
                    unfold_feat_s = unfold_feat_s.resize(4, 512, 16, 16, 16)
                    

                    unfold_feat_t = F.unfold(feat_t, (16, 16), stride=16)
                    unfold_feat_t = unfold_feat_t.resize(4, 512, 16, 16, 16)
                    feat_s = unfold_feat_s
                    feat_t = unfold_feat_t
                #only 128 sp_new_1
                if feat_size_ori == 128:
                    #print(feat_s.size())
                    unfold_feat_s = F.unfold(feat_s, (32, 32), stride=32)
                    unfold_feat_s = unfold_feat_s.resize(4, 256, 32, 32, 16)
                    

                    unfold_feat_t = F.unfold(feat_t, (32, 32), stride=32)
                    unfold_feat_t = unfold_feat_t.resize(4, 256, 32, 32, 16)
                    feat_s = unfold_feat_s
                    feat_t = unfold_feat_t
                
                """feat_anc = torch.cat([feat_s_anc, feat_t_anc])
                feat_cons = torch.cat([feat_s_cons, feat_t_cons])
                feat_anc = pool_dict[feat_size](feat_anc)
                feat_cons = pool_dict[feat_size](feat_cons)
                ind = int(math.log2(512 // feat_s_anc.size(1)))
                #feat_anc = pool_dict[feat_size](feat_anc)
                #feat_cons = pool_dict[feat_size](feat_cons)
                feat_anc = F.normalize(feat_anc, dim=1)
                feat_anc = Avg(feat_anc, ind)
                #print(feat_anc[0, :, 1, 1])
                feat_cons = F.normalize(feat_cons, dim=1)

                corr = (feat_cons * feat_anc).sum(dim=1)
                corr_list = torch.chunk(corr, 2, dim=0)
                #print(corr.size())
                
                loss_sp_inter_sim += Loss_fn(corr_list[0], corr_list[1])"""
                
                for j in range(args.feat_const_batch - 1):
                    anc_feat_s = feat_s[j].unsqueeze(0)
                    anc_feat_t = feat_t[j].unsqueeze(0)
                    cons_feat_s = feat_s[j+1:]
                    cons_feat_t = feat_t[j+1:]
                    repeat_num = args.feat_const_batch - 1 - j
                    if repeat_num > 1:
                        if feat_size_ori >= 64:
                            anc_feat_s = anc_feat_s.repeat(repeat_num, 1, 1, 1, 1)
                            anc_feat_t = anc_feat_t.repeat(repeat_num, 1, 1, 1, 1)
                        else:
                            anc_feat_s = anc_feat_s.repeat(repeat_num, 1, 1, 1)
                            anc_feat_t = anc_feat_t.repeat(repeat_num, 1, 1, 1)
                    anc_feat = torch.cat([anc_feat_s, anc_feat_t])
                    cons_feat = torch.cat([cons_feat_s, cons_feat_t])
                    if feat_size_ori >= 64:
                        anc_feat = anc_feat.permute(0, 4, 1, 2, 3)
                        anc_feat_size = anc_feat.size()
                        anc_feat = anc_feat.reshape(anc_feat_size[0]*anc_feat_size[1], anc_feat_size[2], anc_feat_size[3], anc_feat_size[4])
                        cons_feat = cons_feat.permute(0, 4, 1, 2, 3)
                        cons_feat = cons_feat.reshape(anc_feat_size[0]*anc_feat_size[1], anc_feat_size[2], anc_feat_size[3], anc_feat_size[4])
                    anc_feat = anc_feat.view(anc_feat.size(0), anc_feat.size(1), -1).permute(0, 2, 1)
                    cons_feat = cons_feat.view(cons_feat.size(0), cons_feat.size(1), -1)
                    #if feat_size_ori == 128:
                    #    print(cons_feat.size())
                    corr = anc_feat.matmul(cons_feat)
                    #corr = corr.view(corr.size(0), -1)
                    mult = corr.size(2)
                    corr = F.softmax(corr, dim=2)
                    corr_list = torch.chunk(corr, 2, dim=0)
                    loss_sp_inter_sim += Loss_fn(corr_list[0], corr_list[1]) * (args.feat_const_batch - 1 - j) * mult  
                    #print(loss_sp_inter_sim)
                    #exit()
                    #print(loss_sp_inter_sim)
                    #print(anc_feat.size(), cons_feat.size())
                if feat_size_ori == 128: break
            loss_sp_inter_sim = loss_sp_inter_sim / 6
            #print(loss_sp_inter_sim)
        if args.sp_inter_sim != 'None' and (i-1) % 4 == 0:
            #print(loss_sp_inter_sim)
            #print(loss_sp_inter_sim)
            g_loss = g_loss + loss_sp_inter_sim * 0.4
        loss_dict["g"] = g_loss
        #loss_dict["r1"] = loss_sp_inter_sim
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        # to save up space
        del g_loss, d_loss, fake_img, fake_pred, real_img, real_pred
        if args.self_sim_loss_new != 'None':
            del loss_d, feat, feat_s, feat_t, self_sim, unfold_feat, _ 
        if args.sp_inter_sim != 'None' and (i-1) % 4 == 0:
            del corr, corr_list, feat_target, feat_source, \
                anc_feat_s, anc_feat_t, cons_feat_s, cons_feat_t, anc_feat, cons_feat
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(
                path_batch_size, args.latent, args.mixing, device)
            if args.proj != 'None':
                w = [generator.module.style(item) for item in noise]
            #w = [generator.module.style(item) for item in noise]
                w = [Proj_module.modulate(item) for item in w]
                fake_img, latents = generator(w, input_is_latent=True, return_latents=True)
            else:
                fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema_module, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % args.img_freq == 0:
                with torch.set_grad_enabled(False):
                    g_ema.eval()
                    if args.proj != 'None':
                        w = [g_ema.module.style(sample_z.data)]
            #w = [generator.module.style(item) for item in noise]
                        w = [Proj_module.modulate(item) for item in w]
                        sample, _ = g_ema(w, input_is_latent=True)
                    else:
                        sample, _ = g_ema([sample_z.data])
                    sample_subz, _ = g_ema([sub_region_z.data])
                    utils.save_image(
                        sample,
                        f"%s/{str(i).zfill(6)}.png" % (imsave_path),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    del sample

            if (i % args.save_freq == 0) and (i > 0):
                torch.save(
                    {
                        "g_ema": g_ema.state_dict(),
                        # uncomment the following lines only if you wish to resume training after saving. Otherwise, saving just the generator is sufficient for evaluations

                        #"g": g_module.state_dict(),
                        #"g_s": g_source.state_dict(),
                        #"d": d_module.state_dict(),
                        #"g_optim": g_optim.state_dict(),
                        #"d_optim": d_optim.state_dict(),
                    },
                    f"%s/{str(i).zfill(6)}.pt" % (model_path),
                )

            if i == args.iter - 2:
                torch.save(
                    {
                        "g_ema": g_ema.state_dict(),
                        # uncomment the following lines only if you wish to resume training after saving. Otherwise, saving just the generator is sufficient for evaluations

                        #"g": g_module.state_dict(),
                        #"g_s": g_source.state_dict(),
                        #"d": d_module.state_dict(),
                        #"g_optim": g_optim.state_dict(),
                        #"d_optim": d_optim.state_dict(),
                    },
                    f"%s/final.pt" % (model_path),
                )



if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--iter", type=int, default=5002)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--img_freq", type=int, default=500)
    parser.add_argument("--kl_wt", type=int, default=1000)
    parser.add_argument("--highp", type=int, default=1)
    parser.add_argument("--subspace_freq", type=int, default=4)
    parser.add_argument("--feat_ind", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--feat_const_batch", type=int, default=4)
    parser.add_argument("--n_sample", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feat_res", type=int, default=128)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--subspace_std", type=float, default=0.1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--source_key", type=str, default='church')
    parser.add_argument("--exp", type=str, default=None, required=True)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", dest='augment', action='store_true')
    parser.add_argument("--no-augment", dest='augment', action='store_false')
    parser.add_argument("--augment_p", type=float, default=0.0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--self_sim_loss_new", type=str, default='None')
    parser.add_argument("--sp_inter_sim", type=str, default='None')
    parser.add_argument("--proj", type=str, default='None')
    parser.add_argument("--inter_sim_freq", type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='van_gogh')
    parser.add_argument('--task', type=int, default=10)
    parser.add_argument('--few_shot_batch', type=int, default=4)
        
    args = parser.parse_args()

    torch.manual_seed(1)
    random.seed(1)

    n_gpu = 4
    args.distributed = n_gpu > 1

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    g_source = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    generator = deepcopy(g_source)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    d_source = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    extra = Extra().to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    e_optim = optim.Adam(
        extra.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )


    module_source = ['landscapes', 'red_noise',
                     'white_noise', 'hands', 'mountains', 'handsv2']
    
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        assert args.source_key in args.ckpt
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_source = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass


        #generator.load_state_dict(ckpt["g"], strict=False)
        g_source.load_state_dict(ckpt_source["g_ema"], strict=False)
        generator.load_state_dict(ckpt_source["g_ema"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        #d_source = nn.parallel.DataParallel(d_source)
        #discriminator = nn.parallel.DataParallel(discriminator)
        discriminator.load_state_dict(ckpt["d"])
        d_source.load_state_dict(ckpt_source["d"])

        if 'g_optim' in ckpt.keys():
            g_optim.load_state_dict(ckpt["g_optim"])
        if 'd_optim' in ckpt.keys():
            d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DataParallel(generator)
        g_ema = nn.parallel.DataParallel(g_ema)
        g_source = nn.parallel.DataParallel(g_source)

        discriminator = nn.parallel.DataParallel(discriminator)
        d_source = nn.parallel.DataParallel(d_source)
        extra = nn.parallel.DataParallel(extra)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.data_path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.few_shot_batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")


    train(args, loader, generator, discriminator, extra, g_optim,
          d_optim, e_optim, g_ema, device, g_source, d_source)