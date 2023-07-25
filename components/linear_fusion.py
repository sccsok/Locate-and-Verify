from torch import nn
import torch
from copy import deepcopy

from components.activation_builder import buildActivation
from components.normalization_builder import buildNormalization


class BilinearFusion(nn.Module):

    def __init__(self, seq_dim, img_dim,
                 output_dim,
                 bili_norm_type, bili_affine, bili_non_linear,
                 bili_dropout=None,
                 **kwargs):
        super(BilinearFusion, self).__init__()
        self.Trans = nn.Bilinear(seq_dim, img_dim, output_dim, bias=False)

        if bili_dropout is None:
            self.Dropout = nn.Identity()
        else:
            self.Dropout = nn.Dropout(bili_dropout)

        self.Norm = buildNormalization(norm_name=bili_norm_type,
                                       feature_shape=output_dim,
                                       affine=bili_affine,
                                       norm_name_map={'bn': 'bn_1d',
                                                      'ln': "ln_1d"})

        self.NonLinear = buildActivation(bili_non_linear)

    def forward(self, seq_features, img_features, **kwargs):
        fused_features = self.Trans(seq_features, img_features)
        fused_features = self.Dropout(fused_features)
        fused_features = self.Norm(fused_features)

        return self.NonLinear(fused_features)


class HdmProdBilinearFusion(nn.Module):

    def __init__(self, dim1, dim2,
                 hidden_dim=2048, output_dim=3072, bili_affine=None,
                 bili_dropout=0.5, **kwargs):
        super(HdmProdBilinearFusion, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Trans1 = nn.Linear(dim1, hidden_dim)
        self.Trans2 = nn.Linear(dim2, hidden_dim)
        self.OutTrans = nn.Linear(hidden_dim, output_dim)

        if bili_dropout is None:
            self.Dropout = nn.Identity()
        else:
            self.Dropout = nn.Dropout(bili_dropout)

    def forward(self, features1, features2, **kwargs):
        b1, c1, h1, w1 = features1.size()
        b2, c2, h2, w2 = features2.size()
        assert b1 == b2 and h1 == h2 and w1 == w1
        features1 = features1.view(b1, c1, -1).permute(0, 2, 1).contiguous().view(-1, c1)
        features2 = features2.view(b2, c2, -1).permute(0, 2, 1).contiguous().view(-1, c2)
        prod = self.Trans1(features1) * self.Trans2(features2)
        prod = torch.tanh(prod)
        prod = self.OutTrans(self.Dropout(prod))
        prob = prod.view(b1, -1, self.output_dim).permute(0, 2, 1).contiguous().view(b1, -1, h1, w1)

        return prob


class ResHdmProdBilinearFusion(HdmProdBilinearFusion):
    def __init__(self, seq_dim, img_dim,
                 hidden_dim,                    
                 bili_norm_type, bili_affine,
                 bili_non_linear,
                 bili_dropout=None,
                 **kwargs):
        super_kwargs = deepcopy(kwargs)
        del super_kwargs['output_dim']        
        super(ResHdmProdBilinearFusion, self).__init__(seq_dim, img_dim,
                                                       hidden_dim, seq_dim+img_dim,   
                                                       bili_norm_type, bili_affine,
                                                       bili_non_linear,
                                                       bili_dropout,
                                                       **super_kwargs)

    def forward(self, seq_features, img_features, **kwargs):
        prod = self.SeqTrans(seq_features) * self.ImgTrans(img_features)
        prod = torch.tanh(prod)
        prod = self.Dropout(prod) 
        prod = self.NonLinear(self.Norm(self.OutTrans(prod)))
        cat = torch.cat((seq_features, img_features), dim=1)  
        return prod + cat