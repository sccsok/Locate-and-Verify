import torch
import torch.nn as nn
import torch.nn.functional as F


def sspsl_loss(psegs, pfeas, labels, regions):
    ce_loss = nn.CrossEntropyLoss().to(pfeas.device)
    pfeas = F.normalize(pfeas, dim=1)

    real_anchors = pfeas[labels < 1].permute(0, 2, 3, 1).contiguous().view(-1, pfeas.size(1))
    real_psegs = psegs[labels < 1].permute(0, 2, 3, 1).contiguous().view(-1, psegs.size(1))
    
    fake_anchors = pfeas[labels > 0].permute(0, 2, 3, 1).contiguous().view(-1, pfeas.size(1))
    fake_psegs = psegs[labels > 0].permute(0, 2, 3, 1).contiguous().view(-1, psegs.size(1))

    regions = regions[labels > 0].permute(0, 2, 3, 1).contiguous().view(-1, regions.size(1))

    fake_avg_anchor = torch.sum(fake_anchors * regions, dim=0, keepdim=True) / fake_anchors.size(0)
    real_avg_anchor = torch.sum(real_anchors, dim=0, keepdim=True) / real_anchors.size(0)

    fake_real_sim = torch.cosine_similarity(fake_anchors, real_avg_anchor)
    fake_fake_sim = torch.cosine_similarity(fake_anchors, fake_avg_anchor)

    real_labels = torch.zeros(real_anchors.size(0)).type(torch.LongTensor).to(labels.device)
    fake_labels = (fake_real_sim < fake_fake_sim).type(torch.LongTensor).to(labels.device)
    # fake_labels[fake_sim > (torch.max(real_sim) + torch.min(real_sim)) / 2] = 0
    targets = torch.cat((real_labels, fake_labels), dim=0)
    inputs = torch.cat((real_psegs, fake_psegs), dim=0)
    pct_loss = ce_loss(inputs, targets)
    
    return pct_loss, targets