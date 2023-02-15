import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from criteria.lpips.lpips import LPIPS


class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None, adv_weight=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        self.adv_weight = adv_weight
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()
    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()
    def loss(self, input, target_is_real):
        # CrossEntropy Loss
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        # Least Square Loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        # Hinge Loss
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = torch.mean(torch.relu(1 - input))
            else:
                loss = torch.mean(torch.relu(1 + input))
            return loss
        # Wgan Loss
        else:
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()
    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return (loss / len(input)) * self.adv_weight
        else:
            return (self.loss(input, target_is_real)) * self.adv_weight


class AttLoss(nn.Module):
    def __init__(self, att_weight):
        super(AttLoss, self).__init__()
        self.att_weight = att_weight
        self.criterion = nn.MSELoss()
    def forward(self, att_input, att_output):
        assert len(att_input) == len(att_output), 'attributes maps are not equal'
        att_loss = 0
        for i in range(len(att_input)):
            att_loss += self.criterion(att_input[i], att_output[i])
        att_loss *= 0.5
        return att_loss * self.att_weight


class IdLoss(nn.Module):
    def __init__(self, id_weight, loss_mode):
        super(IdLoss, self).__init__()
        self.id_weight = id_weight
        if loss_mode == 'MSE':
            self.criterion = nn.MSELoss()
            self.mode = loss_mode
        elif loss_mode == 'MAE':
            self.criterion = nn.L1Loss()
            self.mode = loss_mode
        elif loss_mode == 'Cos':
            self.mode = loss_mode
        else:
            raise ValueError('Unexpected Loss Mode {}'.format(loss_mode))
    def forward(self, id_input, id_output):
        if self.mode == 'MSE':
            id_loss = self.criterion(id_input, id_output)
            #id_loss *= 0.5
        elif self.mode == 'MAE':
            id_loss = self.criterion(id_input, id_output)
        elif self.mode == 'Cos':
            id_loss = torch.mean(1 - torch.cosine_similarity(id_input, id_output, dim=1))
        return id_loss * self.id_weight


class RecLoss(nn.Module):
    def __init__(self, rec_weight, loss_mode, device):
        super(RecLoss, self).__init__()
        self.rec_weight = rec_weight
        self.mode = loss_mode
        if self.mode == 'l2':
            self.criterion = nn.MSELoss().to(device)
        elif self.mode == 'lpips':
            self.criterion = LPIPS(net_type='alex').to(device).eval()
        else:
            raise ValueError('Unexpected Loss Mode {}'.format(loss_mode))            
    def forward(self, img_input, img_output):
        if self.mode == 'l2':
            rec_loss = 0.5 * self.criterion(img_input, img_output)
        elif self.mode == 'lpips':
            rec_loss = self.criterion(img_input, img_output)
        return rec_loss * self.rec_weight


class WatLoss(nn.Module):
    def __init__(self, wat_weight, loss_mode):
        super(WatLoss, self).__init__()
        self.wat_weight = wat_weight
        if loss_mode == 'MSE':
            self.criterion = nn.MSELoss()
            self.mode = loss_mode
        elif loss_mode == 'MAE':
            self.criterion = nn.L1Loss()
            self.mode = loss_mode
        elif loss_mode == 'Cos':
            self.mode = loss_mode
        else:
            raise ValueError('Unexpected Loss Mode {}'.format(loss_mode))
    def forward(self, id_output, id_org, wat_ori):
        wat_res = id_output - id_org
        wat_ori = wat_ori.repeat(wat_res.shape[0], 1)

        if self.mode == 'MSE':
            wat_loss = self.criterion(wat_res, wat_ori)
        elif self.mode == 'MAE':
            wat_loss = self.criterion(wat_res, wat_ori)
        elif self.mode == 'Cos':
            wat_loss = torch.mean(1 - torch.cosine_similarity(wat_res, wat_ori, dim=1))
        return wat_loss * self.wat_weight


