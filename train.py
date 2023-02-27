import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None

from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

from loss import MaskLoss, FineMaskLoss
from model import Generator, Discriminator

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
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
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

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

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

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


def concat_noise(global_code, noise):
    out = []
    for n in noise:
        out.append(torch.cat([global_code[0], n], dim=-1))
    return out


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def pixel_norm(input):
    return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        os.makedirs(f"{args.save_path}/res/{args.expname}/images/", exist_ok=True)
        os.makedirs(f"{args.save_path}/res/{args.expname}/ckpts/", exist_ok=True)

        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    g_loss_val = d_loss_val = r1_loss = m_g_loss = m_d_loss = torch.tensor(.0, device=device)
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    loss_m = MaskLoss(min_coverage=args.min_mask, coef_binary=args.coef_binary, coef_area=args.coef_area, min_coef_binary=args.min_coef_binary)
    loss_m_fine = FineMaskLoss(max_coverage=args.max_mask_fine, coef_m_fine=args.coef_m_fine)

    accumulate(g_ema, g_module, 0)

    num_fg_only = args.batch // (args.fg_ratio + 1) * args.fg_ratio
    num_bg = args.batch - num_fg_only

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # sample noise vectors
        noise = mixing_noise(args.batch, args.latent // (args.ch_ratio + 1) * (args.ch_ratio), args.mixing, device)
        noise_bg = [noise[0][:num_bg, :args.latent // ((args.ch_ratio + 1))]]

        fg, bg, pre_m, _, _, refine_m = generator([noise, noise_bg])

        # fade in the refine mask
        if i < args.fade_in:
            refine_m = refine_m * (i / args.fade_in)

        # clip the final mask
        m = torch.clamp(pre_m[:num_bg] + refine_m[:num_bg], min=0, max=1)

        # composite
        fake_img = fg[:num_bg] * m + bg * (1-m)
        fake_img = torch.cat([fake_img, fg[num_bg:]], dim=0)

        fake_pred, m_fake_pred = discriminator(fake_img)
        m_fake_fg_pred = discriminator(fg[:num_bg], out_m_only=True)

        real_pred, _ = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        # mask prediction loss
        if args.use_mask_pred:
            m = F.interpolate(m, size=16, mode='bilinear')
            m_d_loss = 0.5 * F.mse_loss(m, m_fake_pred[:num_bg]) + 0.5 * F.mse_loss(m, m_fake_fg_pred)
            d_loss = d_loss + m_d_loss

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        ##############################

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            real_pred, _ = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        ##############################

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent // (args.ch_ratio + 1) * (args.ch_ratio), args.mixing, device)
        noise_bg = [noise[0][:num_bg, :args.latent // ((args.ch_ratio + 1))]]

        fg, bg, pre_m, _, _, refine_m = generator([noise, noise_bg], track_ema=True)

        if i < args.fade_in:
            refine_m = refine_m * (i / args.fade_in)

        # clamp final mask
        m = torch.clamp(pre_m[:num_bg] + refine_m[:num_bg], min=0, max=1)

        # composite
        fake_img = fg[:num_bg] * m + bg * (1-m)
        fake_img = torch.cat([fake_img, fg[num_bg:]], dim=0)

        fake_pred, m_fake_pred = discriminator(fake_img)

        g_loss = g_nonsaturating_loss(fake_pred)

        # mask consistency loss
        if args.use_mask_pred and i % args.mask_reg_every == 0:
            m_fake_fg_pred = discriminator(fg[:num_bg].detach(), out_m_only=True)
            m_consist_loss = F.mse_loss(m_fake_fg_pred, m_fake_pred[:num_bg])
            g_loss += m_consist_loss

        # mask binarization & mask coverage loss
        m_g_loss = 0.
        m_g_loss += loss_m_fine(torch.clamp(m - pre_m[:num_bg], min=0, max=1))
        m_g_loss += loss_m(pre_m, i, mask=None)

        g_loss += m_g_loss

        # background participation loss
        if i % args.bg_part_reg_every == 0:
            bg_part_loss = F.mse_loss(fg[:num_bg].detach() * m + bg.detach() * (1-m), bg)
            g_loss += bg_part_loss

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"p: {ada_aug_p:.4f}, md: {m_d_loss:.4f}, mg: {m_g_loss:.4f}, bg_part: {bg_part_loss:.4f}, m_consist:{m_consist_loss:.4f}"
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
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                    }
                )

            if i % args.image_every == 0:
                with torch.no_grad():
                    g_ema.eval()

                    noise = mixing_noise(args.batch // 2, args.latent // (args.ch_ratio + 1) * (args.ch_ratio), args.mixing, device)
                    noise_bg = [noise[0][:, :args.latent // ((args.ch_ratio + 1))]]

                    fg, bg, pre_m, _, _, refine_m = g_ema([noise, noise_bg], truncation=args.psi)
                    fg2, bg2, pre_m2, _, _, refine_m2 = g_ema([noise, noise_bg], truncation=1)

                    fg = torch.cat([fg, fg2], dim=0)
                    bg = torch.cat([bg, bg2], dim=0)
                    pre_m = torch.cat([pre_m, pre_m2], dim=0)
                    refine_m = torch.cat([refine_m, refine_m2], dim=0)

                    if i < args.fade_in:
                        refine_m = refine_m * (i / args.fade_in)

                    m = torch.clamp(pre_m + refine_m, min=0, max=1)
                    fake_img = fg*m + bg*(1-m)

                    sample_set = torch.cat(\
                        [fg, ((m-0.5)*2).repeat(1,3,1,1), ((m - pre_m - 0.5)*2).repeat(1,3,1,1), fg*m, bg, fake_img, ], dim=0)

                    del fg2, bg2, pre_m2, refine_m2

                    utils.save_image(
                        sample_set,
                        f"{args.save_path}/res/{args.expname}/images/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % args.save_every == 0 and i != 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"{args.save_path}/res/{args.expname}/ckpts/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=300001, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=8,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=1, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0., help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    #######
    parser.add_argument(
        "--expname", type=str,
        default='ffhq',
    )
    parser.add_argument(
        "--min_mask", type=float,
        default=0.35,
        help="minimum coverage of foreground mask"
    )
    parser.add_argument(
        "--max_mask_fine", type=float,
        default=0.01,
        help="maximum coverage of residual(= final mask - coarse_mask) mask"
    )
    parser.add_argument(
        "--coef_binary", type=float,
        default=5,
        help="coefficient for mask binarization loss"
    )
    parser.add_argument(
        "--min_coef_binary", type=float,
        default=0.5,
        help="minimum coefficient for mask binarization loss (if the mask is opaque, this value may needs to be raised)"
    )
    parser.add_argument(
        "--coef_area", type=float,
        default=5,
        help="coefficient for (coarse) mask area loss"
    )
    parser.add_argument(
        "--coef_m_fine", type=float,
        default=5,
        help="coefficient for (fine) mask inverse area loss"
    )
    parser.add_argument(
        "--save_path", type=str,
        default='.'
    )
    parser.add_argument(
        "--psi", type=float,
        default=0.5,
        help="truncation psi for visualization"
    )
    parser.add_argument(
        "--fg_ratio", type=int,
        default=1,
        help="ratio of number of foreground images in minibatch"
    )
    parser.add_argument(
        "--ch_ratio", type=int,
        default=3,
        help="ratio of number of foreground channels in generator"
    )
    parser.add_argument(
        "--use_mask_pred", action="store_true",
        help="use mask prediction in discriminator for mask consistency regularization"
    )
    parser.add_argument(
        "--mask_reg_every", type=int,
        default=2,
        help="interval of the applying mask consistency regularization"
    )
    parser.add_argument(
        "--bg_part_reg_every", type=int,
        default=2,
        help="interval of the applying background participation regularization"
    )
    parser.add_argument(
        "--image_every", type=int,
        default=500,
        help="interval of visualization"
    )
    parser.add_argument(
        "--save_every", type=int,
        default=10000,
        help="interval of model saving"
    )
    parser.add_argument(
        "--fade_in", type=int,
        default=5000,
        help="length of fade-in the refine mask in the beginning"
    )
    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 2

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, ch_ratio=(args.ch_ratio, 1)
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, use_mask_pred=args.use_mask_pred
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, ch_ratio=(args.ch_ratio, 1)
        ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    #####

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr= args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr= args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            #ckpt_name = os.path.basename(args.ckpt)
            #int(os.path.splitext(ckpt_name)[0])
            args.start_iter = int(args.ckpt[-9:-3]) 
        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])


    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,find_unused_parameters=True
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,find_unused_parameters=True
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)

"""
[for ffhq 256 train (use about 22GB of VRAM)]

CUDA_VISIBLE_DEVICES=0 python -m train --batch=16 --size=256 --expname='ffhq' --min_mask=0.35 --use_mask_pred [FFHQ_LMDB_PATH]
"""