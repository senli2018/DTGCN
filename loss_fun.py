import torch
import torch.nn as nn


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class MMD_loss(torch.nn.Module):

    def __init__(self):
        super(MMD_loss, self).__init__()

    def forward(self, x_src, x_tar):
        return mmd_rbf_noaccelerate(x_src, x_tar)


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, out_1, out_2, Y):
        # torch.FloatTensor(out_1)
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2)
        loss_contrastive = torch.mean(Y * torch.pow(euclidean_distance, 2) +
                                      (1 - Y) * torch.pow
                                      (torch.clamp(self.margin - euclidean_distance.float(), min=0.0), 2))
        return loss_contrastive
