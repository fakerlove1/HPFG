import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from torchvision.transforms import transforms


class Pixel_Class_Contrastive_Loss(nn.Module):
    def __init__(self, num_classes=4):
        super(Pixel_Class_Contrastive_Loss, self).__init__()
        self.num_classes = num_classes

    def contrastiveLoss(self, pos, neg, temperature=0.1):
        """
        :param pos(Tensor): Nx1 positive similarity.
        :param neg(Tensor): Nxk negative similarity.
        :return dict[str, Tensor]:  A dictionary of loss components.
        """
        criterion = nn.CrossEntropyLoss()
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)
        losses = criterion(logits, labels)
        return losses

    # 定义对比学习loss,使得类内分离
    def forward(self, features: torch.Tensor, memory_features, labels, student_predict, teacher_predict):
        '''
        features: [b,c,h,w]
        memory_features: [b,c,h,w]
        labels: [b,h,w]
        student_predict: [b,num_class,h,w]
        teacher_predict: [b,num_class,h,w]
        return:  returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
        '''
        loss = 0.0
        student_predict = torch.argmax(torch.softmax(student_predict, dim=1), dim=1, keepdim=False)  # [b,h,w]
        teacher_predict = torch.argmax(torch.softmax(teacher_predict, dim=1), dim=1, keepdim=False)  # [b,h,w]

        mask_prediction_correctly_student = ((student_predict == labels).float() * (student_predict > 0).float()).bool()
        mask_prediction_correctly_teacher = ((teacher_predict == labels).float() * (teacher_predict > 0).float()).bool()

        features = features.permute(0, 2, 3, 1)  # [b,c,h,w]->[b,h,w,c]
        memory_features = memory_features.permute(0, 2, 3, 1)  # [b,c,h,w]->[b,h,w,c]

        features = features[mask_prediction_correctly_student]  # 选取student预测正确的特征  #[b*h*w,c]
        memory_features = memory_features[mask_prediction_correctly_teacher]  # 选取teacher预测正确的特征  #[b*h*w,c]

        mask_label_correctly_student = student_predict[mask_prediction_correctly_student].reshape(-1)  # [b*h*w]
        mask_label_correctly_teacher = teacher_predict[mask_prediction_correctly_teacher].reshape(-1)  # [b*h*w]

        length = 1024
        oppose_length = 1024*self.num_classes
        for c in range(1, self.num_classes):
            student_mask_c = mask_label_correctly_student == c
            teacher_mask_c = mask_label_correctly_teacher == c
            teacher_oppose_c = mask_label_correctly_teacher != c

            if teacher_mask_c.sum().data < length or teacher_oppose_c.sum() < length or teacher_oppose_c.sum() < oppose_length:
                continue

            features_c = features[student_mask_c, ...][0:length, ...]
            memory_c = memory_features[teacher_mask_c, ...][0:length, ...]
            oppose_c = memory_features[teacher_oppose_c, ...][0:oppose_length, ...]

            features_c = F.normalize(features_c, dim=1)
            memory_c = F.normalize(memory_c, dim=1)
            oppose_c = F.normalize(oppose_c, dim=1).permute(1, 0)

            l_pos = torch.einsum('nc,nc->n', [features_c, memory_c]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [features_c, oppose_c])
            loss += self.contrastiveLoss(l_pos, l_neg, temperature=0.1)

        return loss


if __name__ == '__main__':
    # print(224*224*32/(56*56*8))
    features = torch.randn(8, 32, 224, 224).cuda()
    memory_features = torch.randn(8, 32, 224, 224).cuda()
    labels = torch.randint(low=0, high=4, size=(8, 224, 224)).cuda()

    student_predict = torch.randn(8, 4, 224, 224).cuda().cuda()
    teacher_predict = torch.randn(8, 4, 224, 224).cuda().cuda()

    pixel_loss = Pixel_Contrastive_Loss(num_classes=4).cuda()
    loss = pixel_loss(features, memory_features, labels, student_predict, teacher_predict)
    print(loss)
