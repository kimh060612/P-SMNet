import torch
from torch import nn
from torch.distributions import kl

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

class SemmapLoss(nn.Module):
    def __init__(self):
        super(SemmapLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, obj_gt, obj_pred, mask):
        mask = mask.float()
        loss = self.loss(obj_pred, obj_gt)
        loss = torch.mul(loss, mask)
        # -- mask is assumed to have a least one value
        loss = loss.sum()/mask.sum()
        return loss
    
class AuxSemmapLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(AuxSemmapLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.beta = beta

    def forward(self, obj_gt, obj_pred, mask):
        mask = mask.float()
        # print(obj_gt.shape, torch.argmax(obj_pred, dim=1, keepdim=True).shape)
        pred_ten = torch.argmax(obj_pred, dim=1, keepdim=True).float().squeeze(1)
        loss = self.loss(obj_pred, obj_gt) + self.beta * self.mse_loss(pred_ten, obj_gt.float())
        loss = torch.mul(loss, mask)
        # -- mask is assumed to have a least one value
        loss = loss.sum()/mask.sum()
        return loss

class PSEMMapLoss(nn.Module):
    def __init__(self, beta=10.0):
        super(PSEMMapLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)
        self.beta = beta
        
    def forward(self, map_gt, mask, recon_sample, posterior_l, prior_l, val=False):
        if not val:
            mask = mask.float()
            kl_loss = kl.kl_divergence(posterior_l, prior_l)
            recon_loss = self.criterion(input=recon_sample, target=map_gt)
            reconstruction_loss = torch.mean(torch.mul(recon_loss, mask))
            return -(reconstruction_loss + self.beta * kl_loss)
        else:
            mask = mask.float()
            loss = self.loss(recon_sample, map_gt)
            loss = torch.mul(loss, mask)
            # -- mask is assumed to have a least one value
            loss = loss.sum()/mask.sum()
            return loss
