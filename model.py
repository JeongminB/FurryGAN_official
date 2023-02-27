import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


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
        self.register_buffer("kernel", kernel)

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
        self.register_buffer("kernel", kernel)

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

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, transposed=False
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.transposed = transposed

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        if self.transposed:
            out = conv2d_gradfix.conv_transpose2d(
                input,
                self.weight.transpose(0, 1) * self.scale,
                bias=self.bias,
                stride=2,
                padding=0,
            )
            return out

        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
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
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


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
        fused=True,
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
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

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
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


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

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
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
        ch_ratio=(3,1),
        n_latent4avg = 1024,
    ):
        super().__init__()

        self.size = size
        self.n_latent4avg = n_latent4avg
        self.style_dim = style_dim
        self.ch_ratio = ch_ratio
        style_dim_fg = (style_dim // sum(ch_ratio)) * ch_ratio[0]
        style_dim_bg = (style_dim // sum(ch_ratio)) * ch_ratio[1]
        self.style_dim_fg = style_dim_fg
        self.style_dim_bg = style_dim_bg

        # foreground mapping network
        layers = [PixelNorm()]
        layers.append(
            EqualLinear(style_dim_fg, style_dim_fg, lr_mul=lr_mlp, activation="fused_lrelu")
        )
        for i in range(n_mlp - 1):
            layers.append(
                EqualLinear(style_dim_fg, style_dim_fg, lr_mul=lr_mlp, activation="fused_lrelu")
            )
        self.style_fg = nn.Sequential(*layers)

        # background mapping network
        layers = [PixelNorm()]
        layers.append(
            EqualLinear(style_dim_bg, style_dim_bg, lr_mul=lr_mlp, activation="fused_lrelu")
        )
        for i in range(n_mlp - 1):
            layers.append(
                EqualLinear(style_dim_bg, style_dim_bg, lr_mul=lr_mlp, activation="fused_lrelu")
            )
        self.style_bg = nn.Sequential(*layers)

        self.channels_fg = {
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
        self.channels_bg = self.channels_fg.copy()
        self.channels_fg.update((key, value // sum(ch_ratio) * ch_ratio[0]) for key, value in self.channels_fg.items())
        self.channels_bg.update((key, value // sum(ch_ratio) * ch_ratio[1]) for key, value in self.channels_bg.items())     

        self.input_fg = ConstantInput(self.channels_fg[4])
        self.input_bg = ConstantInput(self.channels_bg[4])

        self.conv1_fg = StyledConv(
            self.channels_fg[4], self.channels_fg[4], 3, style_dim_fg, blur_kernel=blur_kernel
        )
        self.conv1_bg = StyledConv(
            self.channels_bg[4], self.channels_bg[4], 3, style_dim_bg, blur_kernel=blur_kernel
        )

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs_fg = nn.ModuleList()
        self.convs_bg = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs_fg = nn.ModuleList()
        self.to_rgbs_bg = nn.ModuleList()
        self.noises = nn.Module()

        in_channel_fg = self.channels_fg[4]
        in_channel_bg = self.channels_bg[4]

        self.margin = 0.
        self.ema = 0.99

        latent_in = torch.randn(n_latent4avg, self.style_dim_fg)
        self.ema_latent_fg = self.style_fg(latent_in).mean(0, keepdim=False).to('cuda')

        latent_in = torch.randn(n_latent4avg, self.style_dim_bg)
        self.ema_latent_bg = self.style_bg(latent_in).mean(0, keepdim=False).to('cuda')


        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel_fg = self.channels_fg[2 ** i]
            out_channel_bg = self.channels_bg[2 ** i]

            self.convs_fg.append(StyledConv(in_channel_fg, out_channel_fg, 3, style_dim_fg, upsample=True, blur_kernel=blur_kernel,))
            self.convs_fg.append(StyledConv(out_channel_fg, out_channel_fg, 3, style_dim_fg, blur_kernel=blur_kernel,))
            self.convs_bg.append(StyledConv(in_channel_bg, out_channel_bg, 3, style_dim_bg, upsample=True, blur_kernel=blur_kernel,))
            self.convs_bg.append(StyledConv(out_channel_bg, out_channel_bg, 3, style_dim_bg, blur_kernel=blur_kernel,))

            in_channel_fg = out_channel_fg
            in_channel_bg = out_channel_bg
        self.to_fg = ToRGB(out_channel_fg, style_dim_fg)
        self.to_bg = ToRGB(out_channel_bg, style_dim_bg)

        self.n_latent = self.log_size * 2 - 2
        self.pre_m = StyledConv(in_channel_fg, 32, 1, style_dim_fg, upsample=False, blur_kernel=blur_kernel,)
        self.refine_m = StyledConv(in_channel_fg, 32, 1, style_dim_fg, upsample=False, blur_kernel=blur_kernel,)

    def make_noise(self):
        device = self.input_fg.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self):
        latent_in = torch.randn(self.n_latent4avg, self.style_dim_fg).to(self.input_fg.input.device)
        self.ema_latent_fg = self.style_fg(latent_in).mean(0, keepdim=False).to(self.input_fg.input.device)

        latent_in = torch.randn(self.n_latent4avg, self.style_dim_bg).to(self.input_fg.input.device)
        self.ema_latent_bg = self.style_bg(latent_in).mean(0, keepdim=False).to(self.input_fg.input.device)
        return self.ema_latent_fg, self.ema_latent_bg

    def get_latent(self, input, fg=False, bg=False):
        w_fg = self.style_fg(input) if fg else None
        w_bg = self.style_bg(input) if bg else None
        return w_fg, w_bg

    def forward(
        self,
        styles,
        inject_index=None,
        replace_idx=None,
        truncation=1,
        truncation_latent_fg=None,
        truncation_latent_bg=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        fg=True,
        bg=True,
        bg_only=False,
        track_ema=False,
    ):
        if not input_is_latent:
            if fg:
                styles_fg = [self.style_fg(s) for s in styles[0]]
            if bg:
                styles_bg = [self.style_bg(s) for s in styles[1]]
        else:
            if fg:
                styles_fg = [styles[0]]
            if bg:
                styles_bg = [styles[1]]

        if self.training and track_ema:
            with torch.no_grad():
                self.ema_latent_fg = self.ema_latent_fg * (self.ema) + styles_fg[0][0,:] * (1 - self.ema) if fg else self.ema_latent_fg
                self.ema_latent_bg = self.ema_latent_bg * (self.ema) + styles_bg[0][0,:] * (1 - self.ema) if bg else self.ema_latent_bg

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            if fg:
                truncation_latent_fg = self.ema_latent_fg
                style_t_fg = []
                for style_fg in styles_fg:
                    style_t_fg.append(
                        truncation_latent_fg + truncation * (style_fg - truncation_latent_fg)
                    )
                styles_fg = style_t_fg

            if bg:
                truncation_latent_bg = self.ema_latent_bg
                style_t_bg = []
                for style_bg in styles_bg:
                    style_t_bg.append(
                        truncation_latent_bg + truncation * (style_bg - truncation_latent_bg)
                    )
                styles_bg = style_t_bg

        if fg and replace_idx is not None:
            latent_fg = []
            latent_fg.append(styles_fg[0][0].repeat(self.n_latent, 1).unsqueeze(0))
            latent_fg.append(styles_fg[0][1].repeat(self.n_latent, 1).unsqueeze(0))
            for idx_set in replace_idx:
                latent_fg_tmp = styles_fg[0][0].repeat(self.n_latent, 1)
                for idx in idx_set:
                    latent_fg_tmp[idx] = styles_fg[0][1]
                latent_fg.append(latent_fg_tmp.unsqueeze(0))
            latent_fg = torch.cat(latent_fg, dim=0)

        elif fg and len(styles_fg) < 2:
            inject_index = self.n_latent
            if styles_fg[0].ndim < 3:
                latent_fg = styles_fg[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent_fg = styles_fg[0]
        elif fg:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent_fg = styles_fg[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2_fg = styles_fg[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent_fg = torch.cat([latent_fg, latent2_fg], 1)

        if bg and replace_idx is not None:
            latent_bg = []
            latent_bg.append(styles_bg[0][0].repeat(self.n_latent, 1).unsqueeze(0))
            latent_bg.append(styles_bg[0][1].repeat(self.n_latent, 1).unsqueeze(0))
            for idx_set in replace_idx:
                latent_bg_tmp = styles_bg[0][0].repeat(self.n_latent, 1)
                for idx in idx_set:
                    latent_bg_tmp[idx] = styles_bg[0][1]
                latent_bg.append(latent_bg_tmp.unsqueeze(0))
            latent_bg = torch.cat(latent_bg, dim=0)

        elif bg and len(styles_bg) < 2:
            inject_index = self.n_latent
            if styles_bg[0].ndim < 3:
                latent_bg = styles_bg[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent_bg = styles_bg[0]

        elif bg:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent_bg = styles_bg[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2_bg = styles_bg[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent_bg = torch.cat([latent_bg, latent2_bg], 1)

        out_fg, out_bg, m, refine_m = None, None, None, None

        if fg:
            out_fg = self.input_fg(latent_fg)
            out_fg = self.conv1_fg(out_fg, latent_fg[:, 0], noise=noise[0])

        if bg:
            out_bg = self.input_bg(latent_bg)
            out_bg = self.conv1_bg(out_bg, latent_bg[:, 0], noise=noise[0])

        if fg and not bg_only:
            i = 1
            for conv1, conv2, noise1, noise2 in zip(
                self.convs_fg[::2], self.convs_fg[1::2], noise[1::2], noise[2::2]
            ):
                out_fg = conv1(out_fg, latent_fg[:, i], noise=noise1)
                out_fg = conv2(out_fg, latent_fg[:, i + 1], noise=noise2)
                i += 2
        
            m = F.relu(self.pre_m(out_fg, latent_fg[:, -1]))
            m = m.mean(dim=1, keepdim=True)
            m_min = torch.min(m.view(m.size(0),-1,1,1), dim=1, keepdim=True)[0]
            m_max = torch.max(m.view(m.size(0),-1,1,1), dim=1, keepdim=True)[0]
            m = (1 + self.margin) * (m - m_min) / (m_max - m_min + 1e-10)

            refine_m = F.relu(self.refine_m(out_fg, latent_fg[:, -1]))
            refine_m = refine_m.mean(dim=1, keepdim=True)
            refine_m_min = torch.min(refine_m.view(refine_m.size(0),-1,1,1), dim=1, keepdim=True)[0]
            refine_m_max = torch.max(refine_m.view(refine_m.size(0),-1,1,1), dim=1, keepdim=True)[0]
            refine_m = (1 + self.margin) * (refine_m - refine_m_min) / (refine_m_max - refine_m_min + 1e-10)

            out_fg = self.to_fg(out_fg, latent_fg[:, -1])
            out_fg = torch.tanh(out_fg)

        if bg:
            i = 1
            for conv1, conv2,  noise1, noise2 in zip(
                self.convs_bg[::2], self.convs_bg[1::2], noise[1::2], noise[2::2]
            ):
                out_bg = conv1(out_bg, latent_bg[:, i], noise=noise1)
                out_bg = conv2(out_bg, latent_bg[:, i + 1], noise=noise2)

                i += 2

            out_bg = self.to_bg(out_bg, latent_bg[:, -1])
            out_bg = torch.tanh(out_bg)

        return out_fg, out_bg, m, latent_fg, latent_bg, refine_m


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
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

        transposed = True if upsample else False
        
        if not upsample:
            layers.append(nn.ReflectionPad2d(self.padding))

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=0,
                transposed=transposed,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True, k=3):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, k)
        self.conv2 = ConvLayer(in_channel, out_channel, k, downsample=downsample)

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
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], use_mask_pred=False):
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

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channels[size], 1))

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

        self.use_mask_pred = use_mask_pred
        if use_mask_pred:
            self.conv_m = ResBlock(channels[16], 32, blur_kernel, downsample=False, k=1)

    def forward(self, input, pred_w_bg=False, out_m_only=False, no_grad=False):
        out = input
        m_pred = None

        for m in self.convs:
            out = m(out)
            if out.size(-1) == 16:
                if self.use_mask_pred:
                    m_pred = (F.relu(self.conv_m(out))).mean(dim=1, keepdim=True)
                    m_pred_min = torch.min(m_pred.view(m_pred.size(0),-1,1,1), dim=1, keepdim=True)[0]
                    m_pred_max = torch.max(m_pred.view(m_pred.size(0),-1,1,1), dim=1, keepdim=True)[0]
                    m_pred = (m_pred - m_pred_min) / (m_pred_max- m_pred_min + 1e-10)
                    if out_m_only:
                        return m_pred

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
        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out, m_pred
