import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CriterionStructuralKD']


class CriterionStructuralKD(nn.Module):
    def __init__(self):
        super(CriterionStructuralKD, self).__init__()

    def pair_wise_sim_map(self, fea):
        B, C, H, W = fea.size()
        fea = fea.reshape(B, C, -1)
        fea_T = fea.transpose(1,2)
        sim_map = torch.bmm(fea_T, fea)
        return sim_map

    def forward(self, feat_S, feat_T):
        # Structured knowledge distillation for semantic segmentation. CVPR-2019.
        B, C, H, W = feat_S.size()

        #feat_S = feat_S.reshape(B, C, -1)
        patch_w = 4 #int(0.5 * W)
        patch_h = 4 #int(0.5 * H)
        maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        
        S_sim_map = self.pair_wise_sim_map(feat_S)
        T_sim_map = self.pair_wise_sim_map(feat_T)
        B, H, W = S_sim_map.size()

        sim_err = ((S_sim_map - T_sim_map)**2)
        sim_dis = sim_err.mean()
        
        return sim_dis
    
class Kld_Loss(nn.Module):
    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        targets = F.softmax(targets, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')

if __name__=="__main__":
    x=torch.randn(32,3,224,224)
    x1=torch.randn(32,3,224,224)

    criterion = CriterionStructuralKD()
    print(criterion(x,x1))


