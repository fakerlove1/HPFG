import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class SimCLRLoss(nn.Module):
    def __init__(self, batch_size:int, device=torch.device('cuda'), temperature=0.5):
        super(SimCLRLoss, self).__init__()

        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.batch_size = batch_size
        self.temperature = temperature


    def forward(self, out_1,out_2):
        out_1 = F.normalize(out_1, dim=1)     # (bs, dim)  --->  (bs, dim)
        out_2 = F.normalize(out_2, dim=1)     # (bs, dim)  --->  (bs, dim)

        # [2*B, D]
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
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss



if __name__ == '__main__':
    loss_func = SimCLRLoss(batch_size=4)
    emb_i = torch.rand(4, 512).cuda()
    emb_j = torch.rand(4, 512).cuda()
    loss_contra = loss_func(emb_i, emb_j)
    print(loss_contra)

   