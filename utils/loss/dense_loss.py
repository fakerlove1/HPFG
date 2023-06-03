import torch.nn as nn
import torch
import torch.nn.functional as F

class Dense_Loss(nn.Module):

    def __init__(self,
                 batch_size: int = 32,
                 device=torch.device('cuda'),
                 temperature=0.7):
        super(Dense_Loss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature
        # self.cretrion = simclr_loss(batch_size, device, temperature)

    def contrastive_loss(self, out_1, out_2):
        out_1 = F.normalize(out_1, dim=1).flatten(1)  # (bs, dim,SxS)  --->  (bs, dim*S*S)
        out_2 = F.normalize(out_2, dim=1).flatten(1)  # (bs, dim,SxS)  --->  (bs, dim*S*S)

        # [2*B, dim,SxS]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def forward(self, x, y):
        x1, x2 = x
        y1, y2 = y
        loss1 = self.contrastive_loss(x1, y1.detach())
        loss2 = self.contrastive_loss(x2, y2.detach())
        return 0.5 * (loss1 + loss2)