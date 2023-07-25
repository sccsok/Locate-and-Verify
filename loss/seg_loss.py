import torch
import torch.nn as nn
import torch.nn.functional as F


class FSCELoss(nn.Module):
    def __init__(self):
        super(FSCELoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, weights=None):

        targets = targets.type(torch.cuda.LongTensor)
        
        n, c, h, w = inputs.size()
        nt, ct, ht, wt = targets.size()

        # Handle inconsistent size between input and target
        if h != ht and w != wt:  # upsample labels
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        # targets = targets.view(-1).type(torch.LongTensor)
        targets = targets.view(-1)

        loss = F.cross_entropy(inputs, targets, weight=weights)

        return loss

