import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np


def min_mask_loss(mask, min_mask_coverage):
    return F.relu(min_mask_coverage - mask.mean(dim=(1, 2, 3))).mean()

def min_permask_loss(mask, min_mask_coverage):
    return F.relu(min_mask_coverage - mask.mean(dim=(2, 3))).mean()

def min_mask_loss_batch(mask, min_mask_coverage):
    return F.relu(min_mask_coverage - mask.mean())

def binarization_loss(mask):
    return torch.min(1-mask, mask).mean()

def create_circle_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask.astype(int)


class MaskLoss:
    def __init__(self, min_coverage=0.35, coef_binary=5, coef_area=5, min_coef_binary=0.5, decay_length=5000, device='cpu'):
        self.coef_binary = coef_binary
        self.coef_area = coef_area
        self.phi1 = min_coverage
        self.min_coef_binary = min_coef_binary
        self.decay_length = decay_length

    def __call__(self, M, current_iter, mask=None):
        if mask == None:
            loss_binary = torch.min(1-M, M).mean()
        else:
            loss_binary = (torch.min(1-M, M)*mask).mean()

        loss_area = F.relu(self.phi1 - M.mean(dim=(-2,-1))).mean()

        return max((self.min_coef_binary - self.coef_binary) / self.decay_length * current_iter + self.coef_binary, self.min_coef_binary) * loss_binary \
            + self.coef_area * loss_area


class FineMaskLoss:
    def __init__(self, min_coverage=0.0, max_coverage=0.01, coef_m_fine=5, device='cpu'):
        self.coef_m_fine = coef_m_fine
        self.phi1 = min_coverage
        self.phi2 = 1 - max_coverage

    def __call__(self, M):
        loss_area = F.relu(self.phi1 - M.mean(dim=(-2,-1))).mean()
        loss_inv_area = F.relu(self.phi2 - (1 - M).mean(dim=(-2,-1))).mean()
        return self.coef_m_fine * (loss_area + loss_inv_area)
