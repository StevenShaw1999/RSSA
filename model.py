import math
import random
import functools
import operator
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Upsample as inbuilt_upsample
from torch.autograd import Function
import numpy as np
from torch.nn.parameter import Parameter
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from torch.nn import init
from packaging import version
from functools import reduce


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class avg_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.new_conv = nn.ModuleList()
        conv = nn.Conv2d(in_channels=512, out_channels=512, stride=1, padding=1, bias=False, kernel_size=3)
        conv.weight = nn.Parameter(torch.eye(512).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3) / 9,requires_grad=False)
        self.new_conv.append(conv)
        conv = nn.Conv2d(in_channels=256, out_channels=256, stride=1, padding=1, bias=False, kernel_size=3)
        conv.weight = nn.Parameter(torch.eye(256).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3) / 9,requires_grad=False)
        self.new_conv.append(conv)
        conv = nn.Conv2d(in_channels=128, out_channels=128, stride=1, padding=1, bias=False, kernel_size=3)
        conv.weight = nn.Parameter(torch.eye(128).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3) / 9,requires_grad=False)
        self.new_conv.append(conv)
        conv = nn.Conv2d(in_channels=64, out_channels=64, stride=1, padding=1, bias=False, kernel_size=3)
        conv.weight = nn.Parameter(torch.eye(64).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3) / 9,requires_grad=False)
        self.new_conv.append(conv)

    def forward(self, x, ind):
        #print(self.new_conv[ind].weight)
        #exit()
        return self.new_conv[ind](x)

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            #print('here')
            #print(self.weight)
            batch, _, height, width = image.shape   
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise
        #else:
        #    print('Noise')


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, cons=False):
        out = self.conv(input, style)
        if not cons:
            out = self.noise(out, noise=noise)
        # out = out + self.bias
        #print(out.size())
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None, proj=None):
        if proj is not None:
            input = proj(input)
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        return_feats=False,
        return_hl_feats=False,
        truncation_layers=-1,
        basecode=None,
        basecode_size=-1,
        cons=False,
        proj_module=None,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]
        else:
            inject_index = self.n_latent
            styles_new = []
            for s in styles:
                styles_new.append(s)
            #styles = [s.unsqueeze(1).repeat(1, inject_index, 1) for s in styles if len(s.size())!= 3 else s] 
            styles = styles_new
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for num, style in enumerate(styles):
                if num < truncation_layers:
                    style_t.append(
                        truncation_latent + truncation * (style - truncation_latent)
                    )
                else:
                    style_t.append(style)

            styles = style_t
            latent = styles[0]
        
        else:

            if len(styles) < 2:
                inject_index = self.n_latent
                if styles[0].ndim < 3:
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                else:
                    latent = styles[0]


            else:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)
                #print(styles[0].size())
                if styles[0].ndim < 3:
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                    latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
                    latent = torch.cat([latent, latent2], 1)
                else:
                    latent = torch.cat([styles[0][:,:inject_index, :], styles[1][:,inject_index:, :]], 1)

            #latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
            #latent[:, inject_index-1, :] = styles[1]

        feat_list = []
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0], cons=cons)
        if return_feats:
            feat_list.append(out)
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            
            out = conv1(out, latent[:, i], noise=noise1, cons=cons)
            if return_feats:
                feat_list.append(out)
            elif return_hl_feats:
                if out.size(2) >= 64:
                    feat_list.append(out)
            out = conv2(out, latent[:, i + 1], noise=noise2, cons=cons)
            if return_feats:
                feat_list.append(out)
            elif return_hl_feats:
                if out.size(2) >= 64 and out.size(2) < 256:
                    feat_list.append(out)
            if proj_module is not None:
                skip = to_rgb(out, latent[:, i + 2], skip, getattr(proj_module, f'proj_{out.size(2)}'))
            else:
                skip = to_rgb(out, latent[:, i + 2], skip)
            if basecode is not None and basecode_size == out.size(3):
                #print(out.size(), basecode.size())
                out = basecode.contiguous()
                skip = to_rgb(out, latent[:, i + 2], None)
                if return_feats:
                    feat_list = []


            i += 2

        image = skip
        if return_latents:
            return image, latent
        
        elif return_feats:
            return image, feat_list
        elif return_hl_feats:
            return image, feat_list
        
        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )




    def forward(self, inp, ind = None, real = False):

        feat = []
        for i in range(len(self.convs)):
            if i == 0:
                inp = self.convs[i](inp)
                feat.append(inp)
            else:
                temp1 = self.convs[i].conv1(inp)
                feat.append(temp1)
                temp2 = self.convs[i].conv2(temp1)
                feat.append(temp2)
                inp = self.convs[i](inp)

        out = inp

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)


        out = self.final_conv(out)
        feat.append(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out, feat


class Patch_Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )


    def forward(self, inp, ind = None, extra = None, flag = None, p_ind = None, real=False, dual=False, return_feats=False):

        feat = []
        if return_feats:
            feat_list = []
        for i in range(len(self.convs)):
            if i == 0:
                inp = self.convs[i](inp)
            else:
                temp1 = self.convs[i].conv1(inp)
                if (flag > 0) and (temp1.shape[1] == 512) and (temp1.shape[2] == 32 or temp1.shape[2] == 16):
                    feat.append(temp1)
                temp2 = self.convs[i].conv2(temp1)
                if (flag > 0) and (temp2.shape[1] == 512) and (temp2.shape[2] == 32 or temp2.shape[2] == 16):
                    feat.append(temp2)
                inp = self.convs[i](inp)
                if (flag > 0) and len(feat) == 4:
                    # We use 4 possible intermediate feature maps to be used for patch-based adversarial loss. Any one of them is selected randomly during training.
                    inp_e = extra(feat[p_ind], p_ind)
                    if not dual:
                        return inp_e, None
            if return_feats:
                if inp.size(2) <= 128:
                    feat_list.append(inp)
        if return_feats:
            return feat_list
        out = inp
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        feat.append(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)
        if not dual:
            return out, None 
        else:
            return out, inp_e



class Extra(nn.Module):
    # to apply the patch-level adversarial loss, we take the intermediate discriminator feature maps of size [N x N x D], and convert them into [N x N x 1]

    def __init__(self):
        super().__init__()

        self.new_conv = nn.ModuleList()
        self.new_conv.append(ConvLayer(512, 1, 3))
        self.new_conv.append(ConvLayer(512, 1, 3))
        self.new_conv.append(ConvLayer(512, 1, 3))
        self.new_conv.append(ConvLayer(512, 1, 3))

    def forward(self, inp, ind):
        out = self.new_conv[ind](inp)
        return out


class Smooth_L1_Loss_margin(nn.Module):
    def __init__(self, margin_l, margin_h):
        super().__init__()
        self.margin_l = margin_l
        self.margin_h = margin_h
        self.SML = nn.SmoothL1Loss()

    
    def forward(self, x, y):
        mask = (torch.abs(x - y) > self.margin_l)
        #sz_lis = x.size()
        #red = reduce(lambda t1, t2:t1 * t2,sz_lis)
        return self.SML(x * mask, y * mask)


def get_gaussian_kernel(kernel_size=5, sigma=0.55, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

class Encoder(nn.Module):
    # to apply the patch-level adversarial loss, we take the intermediate discriminator feature maps of size [N x N x D], and convert them into [N x N x 1]

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleDict()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        for i in range(3):
            self.convs.append(nn.Conv2d(512, 512, kernel_size=1))

        self.layers['final'] = nn.Conv2d(512, 512, kernel_size=1)
        self.layers['act'] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layers['pool'] = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, inp_list):
        #out = 0
        for num, item in enumerate(inp_list):
            #sz = str(item.size(2))
            if num == 0:
                out = self.convs[num](item)
            else:
                out = out + item
                out = self.convs[num](out)
            out = self.layers['act'](out)
            if num == 3:
                out = self.layers['final'](out)
            else:
                out = self.layers['pool'](out)
        
        return out

class Proj_func(nn.Module):
    def __init__(self, channel):
        super().__init__()
        t = torch.eye(channel)
        self.proj_mat = nn.Parameter(t)
        self.proj_beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, input):
        input = input.permute(0,2,3,1)
        output = input - self.proj_beta * torch.matmul(input, self.proj_mat)
        output = output.permute(0,3,1,2)
        return output
        

class Inject_proj(nn.Module):
    def __init__(self):
        super().__init__()
        #resolist = [8, 16, 32, 64, 128, 256]
        self.proj_8 = Proj_func(512)
        self.proj_16 = Proj_func(512)
        self.proj_32 = Proj_func(512)
        self.proj_64 = Proj_func(512)
        self.proj_128 = Proj_func(256)
        self.proj_256 = Proj_func(128)

class Projection_module():
    def __init__(self, args):
        super().__init__()

        latent_dir = args.latent_dir
        if args.task == 10:
            if args.exp_name == 'VanGogh':
                choose_idx = range(9)
            else:
                choose_idx = range(10)
        if args.task == 5:
            if args.exp_name == 'sketches': choose_idx = [0, 2, 5, 6, 8]
            elif args.exp_name == 'caricatures': choose_idx = [1, 3, 4, 6, 8]
            elif args.exp_name == 'VanGogh': choose_idx = [1,2,3,4,7]
            else: choose_idx = [0, 1, 2, 3, 4]

        catt = []
        for i in choose_idx:
            arr = np.load(f'%s{i}_latentcode.npy' % latent_dir)
            catt.append(arr)

        catt = torch.from_numpy(np.concatenate(catt)).cuda()
        self.catt_ori = catt.permute(1, 2, 0)
        self.catt_t = catt.permute(1, 0, 2)
        self.inv_mat = torch.inverse(torch.matmul(self.catt_t, self.catt_ori))
        self.compute_mat = self.inv_mat.matmul(self.catt_t).unsqueeze(0)
        sub = torch.zeros(1, 14, 1).cuda() 

        for i in range(3, 5):
            sub[:, i, :] += 0.05
        for i in range(5, 7):
            sub[:, i, :] += 0.05
        for i in range(7, 9):
            sub[:, i, :] += 0.45
        for i in range(9, 11):
            sub[:, i, :] += 0.65
        for i in range(11, 14):
            sub[:, i, :] += 0.85    

        self.sub_ori = sub
        self.sub = sub

    def modulate(self, inp):
        if inp.ndim < 3:
            inp = inp.unsqueeze(1).repeat(1, 14, 1)
        alpha = self.compute_mat.matmul(inp.unsqueeze(-1))
        orth = self.catt_ori.unsqueeze(0).matmul(alpha).squeeze(-1)
        orth = orth / torch.norm(orth, dim=2, keepdim=True) * torch.norm(inp, dim=2, keepdim=True)
        inp = self.sub_ori * orth + (1-self.sub_ori) * inp
        return inp


class Projection_module_diff(nn.Module):
    def __init__(self):
        super().__init__()
        catt = []
        for i in range(10):
            arr = np.load(f'exps/{i}/latentcode.npy')
            catt.append(arr)
        catt = torch.from_numpy(np.concatenate(catt)).cuda()

        self.catt_ori = catt.permute(1, 2, 0)
        self.catt_t = catt.permute(1, 0, 2)
        self.inv_mat = torch.inverse(torch.matmul(self.catt_t, self.catt_ori))
        self.compute_mat = self.inv_mat.matmul(self.catt_t).unsqueeze(0)
        self.compute_mat = nn.Parameter(self.compute_mat, requires_grad=True)

        sub = torch.ones(1, 14, 1).cuda() * 0.1
        for i in range(3, 5):
            sub[:, i, :] += 0.15
        for i in range(5, 7):
            sub[:, i, :] += 0.3
        for i in range(7, 9):
            sub[:, i, :] += 0.45
        for i in range(9, 11):
            sub[:, i, :] += 0.6
        for i in range(11, 14):
            sub[:, i, :] += 0.7
        
        self.sub_ori = sub
        self.sub_ori = nn.Parameter(self.sub_ori, requires_grad=False)
        #print(sub)
        #exit()
    
    def adjust_sub(self, total_iter, n_iter):
        self.sub_ori = max(0, (0.2 + 0.8 * (1 - n_iter / total_iter))) * self.sub_ori

    def forward(self, inp):
        if inp.ndim < 3:
            inp = inp.unsqueeze(1).repeat(1, 14, 1)
        alpha = self.compute_mat.matmul(inp.unsqueeze(-1))
        #print(alpha.size(), catt_ori.unsqueeze(0).size())
        orth = self.catt_ori.unsqueeze(0).matmul(alpha).squeeze()

        inp = self.sub_ori * orth + (1-self.sub_ori) * inp
        #print(hh.size())
        return inp


class Projection_module_church():
    def __init__(self, args):
        super().__init__()
        latent_dir = args.latent_dir
        if args.task == 10:
            choose_idx = range(10)
        if args.task == 5:
            if args.exp_name == 'VanGogh': choose_idx = [0, 1, 3, 4, 7]
            elif args.exp_name == 'haunted': choose_idx = [1, 2, 3, 6, 9]

        catt = []
        for i in choose_idx:
            arr = np.load(f'%s/{i}_latentcode.npy'%latent_dir)
            catt.append(arr)
        catt = torch.from_numpy(np.concatenate(catt)).cuda()
        self.catt_ori = catt.permute(1, 2, 0)
        self.catt_t = catt.permute(1, 0, 2)
        self.inv_mat = torch.inverse(torch.matmul(self.catt_t, self.catt_ori))
        self.compute_mat = self.inv_mat.matmul(self.catt_t).unsqueeze(0)
        sub = torch.zeros(1, 14, 1).cuda() 
        
        for i in range(3, 5):
            sub[:, i, :] += 0.05
        for i in range(5, 7):
            sub[:, i, :] += 0.05
        for i in range(7, 9):
            sub[:, i, :] += 0.35
        for i in range(9, 11):
            sub[:, i, :] += 0.55
        for i in range(11, 14):
            sub[:, i, :] += 0.75 
             
        self.sub_ori = sub
        self.sub = sub
    
    def adjust_sub(self, total_iter, n_iter):
        self.sub_ori = max(0, (0.2 + 0.8 * (1 - n_iter / total_iter))) * self.sub

    def modulate(self, inp):
        if inp.ndim < 3:
            inp = inp.unsqueeze(1).repeat(1, 14, 1)
        alpha = self.compute_mat.matmul(inp.unsqueeze(-1))
        orth = self.catt_ori.unsqueeze(0).matmul(alpha).squeeze(-1)
        orth = orth / torch.norm(orth, dim=2, keepdim=True) * torch.norm(inp, dim=2, keepdim=True)
        inp = self.sub_ori * orth + (1-self.sub_ori) * inp
        return inp


class Projection_module_cars():
    def __init__(self, args):
        super().__init__()

        #print(latent_dir)
        catt = []
        #print(hh, args.exp_name, args.task)
        #exit()
        hh = [2]
        for i in hh:
            arr = np.load(f'%s/{i}/latentcode.npy'%'exps/wrecked_cars')
            catt.append(arr)

        self.catt_ori = catt.permute(1, 2, 0)
        self.catt_t = catt.permute(1, 0, 2)
        self.inv_mat = torch.inverse(torch.matmul(self.catt_t, self.catt_ori))
        self.compute_mat = self.inv_mat.matmul(self.catt_t).unsqueeze(0)
#exit() 
        sub = torch.zeros(1, 16, 1).cuda() 
        for i in range(3, 5):
            sub[:, i, :] += 0.05
        for i in range(5, 7):
            sub[:, i, :] += 0.05
        for i in range(7, 9):
            sub[:, i, :] += 0.05
        for i in range(9, 11):
            sub[:, i, :] += 0.15
        for i in range(11, 13):
            sub[:, i, :] += 0.25
        for i in range(13, 16):
            sub[:, i, :] += 0.35

        self.sub_ori = sub
        #print(sub)
        #exit()
    
    def adjust_sub(self, total_iter, n_iter):
        self.sub_ori = max(0, (0.2 + 0.8 * (1 - n_iter / total_iter))) * self.sub

    def modulate(self, inp):
        if inp.ndim < 3:
            inp = inp.unsqueeze(1).repeat(1, 16, 1)
        alpha = self.compute_mat.matmul(inp.unsqueeze(-1))
        #print(alpha.size(), catt_ori.unsqueeze(0).size())
        orth = self.catt_ori.unsqueeze(0).matmul(alpha).squeeze()
        orth = orth / torch.norm(orth, dim=2, keepdim=True) * torch.norm(inp, dim=2, keepdim=True)
        inp = self.sub_ori * orth + (1-self.sub_ori) * inp
        #print(hh.size())
        return inp
    

class HED_Network(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        arguments_strModel = 'bsds500'
        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch', file_name='hed-' + arguments_strModel).items() })
    # end

    def forward(self, tenInput):
        tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
