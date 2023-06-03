"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ['CriterionMiniBatchCrossImagePair']


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CriterionMiniBatchCrossImagePair(nn.Module):
    def __init__(self, temperature=0.7, pooling: int = 56):
        super(CriterionMiniBatchCrossImagePair, self).__init__()
        self.temperature = temperature
        self.pooling = pooling
        if pooling is not None:
            self.avg_pool = nn.AdaptiveAvgPool2d((pooling, pooling))

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)

        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def forward(self, feat_S, feat_T):
        B, C, H, W = feat_S.size()

        if self.pooling is not None:
            feat_S = self.avg_pool(feat_S)
            feat_T = self.avg_pool(feat_T)

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        sim_dis = torch.tensor(0.).cuda()
        for i in range(B):
            for j in range(B):
                s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[j])
                t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[j])

                p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
                p_t = F.softmax(t_sim_map / self.temperature, dim=1)

                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
                sim_dis += sim_dis_
        sim_dis = sim_dis / (B * B)
        return sim_dis


if __name__ == '__main__':
    criterion = CriterionMiniBatchCrossImagePair(temperature=1, pooling=True)
    input = torch.randn(32, 3, 224, 224)
    x2 = torch.rand_like(input)

    output = criterion(input, x2)
    print(output)
    print(output.size())
