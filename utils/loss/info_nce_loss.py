import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable



class Info_Nce_Loss(nn.Module):
    def __init__(self, batch_size, temperature=0.7, device='cuda', n_views=2):
        super(Info_Nce_Loss, self).__init__()

        self.device = torch.device(device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature

    def forward(self, features):
        """
        features: [n_views*batch_size, dim]
        return: loss
        """
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
        
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        loss = self.criterion(logits, labels)
        return loss



if __name__=="__main__":
    loss_func = Info_Nce_Loss(batch_size=4, n_views=2)
    emb_i = torch.rand(8, 512).cuda()
    loss_contra = loss_func(emb_i)
    print(loss_contra)