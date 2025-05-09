import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class NegativeCosineSimLoss(nn.Module):
    def __init__(self):
        super(NegativeCosineSimLoss, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, p1, p2, z1, z2):
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss


class MeanSquaredError(nn.Module):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, labels):
        loss = self.criterion(output, labels)
        return loss


class NtXentLoss(nn.Module):
    def __init__(self):
        super(NtXentLoss, self).__init__()

    def forward(self, out_1, out_2, temperature=0.5, eps=1e-6):
        out_1 = F.normalize(out_1, dim=-1, p=2)
        out_2 = F.normalize(out_2, dim=-1, p=2)

        out_1_dist = out_1
        out_2_dist = out_2

        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss


class ByolLoss(nn.Module):
    def __init__(self):
        super(ByolLoss, self).__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        return loss


class KLLoss(nn.Module):
    """
    KL-Divergence symmetric loss between two distributions
    Used in here for knowledge distillation
    """

    def __init__(self):
        super(KLLoss, self).__init__()
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, zxs, zys, zxt, zyt, temperature=0.1):
        sim_s = self.similarity_f(zxs.unsqueeze(1), zys.unsqueeze(0)) / temperature
        sim_s = F.softmax(sim_s, dim=1)
        sim_t = self.similarity_f(zxt.unsqueeze(1), zyt.unsqueeze(0)) / temperature
        sim_t = F.softmax(sim_t, dim=1)
        loss_s = F.kl_div(sim_s.log(), sim_t.detach(), reduction='batchmean')
        loss_t = F.kl_div(sim_t.log(), sim_s.detach(), reduction='batchmean')
        return loss_s, loss_t


class SimpleDINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)

        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach()

        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()

        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
