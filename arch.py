from fastprogress import master_bar, progress_bar
#import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
#from skimage.util import montage
import pandas as pd
from torch import optim
import re
import torch
from fastai import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn

def gem(x, p=3, eps=1e-5):
    return torch.abs(F.avg_pool2d(x.clamp(min=eps, max=1e4).pow(p), (x.size(-2), x.size(-1))).pow(1./p))
class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(1).expand_as(x)
        return x

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=torch.clamp(self.p, min=0.1), eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



class GeMConst(nn.Module):

    def __init__(self, p=3.74, eps=1e-6):
        super(GeMConst, self).__init__()
        self.p =p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p) + ', ' + 'eps=' + str(
            self.eps) + ')'
class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class PCBRingHead2(nn.Module):
    def __init__(self, num_classes, feat_dim, num_clf = 4, in_feat = 2048, r_init =1.5):
        super(PCBRingHead2,self).__init__()
        self.eps = 1e-10
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.num_clf = num_clf
        self.local_FE_list = nn.ModuleList()
        self.rings =  nn.ParameterList()
        self.total_clf = nn.Sequential( nn.Dropout(p=0.5),
                                      nn.Linear(in_features=feat_dim*num_clf,
                                                out_features=num_classes, bias=True))
        for i in range(num_clf):
            self.rings.append(nn.Parameter(torch.ones(1).cuda()*r_init))
        for i in range(num_clf):
            self.local_FE_list.append(nn.Sequential(GeMConst(3.74), Flatten(),
                        nn.BatchNorm1d(in_feat, eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True),
                        nn.Dropout(p=0.3),
                        nn.Linear(in_features=in_feat, out_features=feat_dim, bias=True),
                        nn.CELU(inplace=True),
                        nn.BatchNorm1d(feat_dim,eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True)
                        ))
        self.local_clf_list = nn.ModuleList()
        for i in range(num_clf):
            self.local_clf_list.append(nn.Sequential( nn.Dropout(p=0.5),
                                      nn.Linear(in_features=feat_dim, out_features=num_classes, bias=True)))
    def forward(self, x):
        assert x.size(3) % self.num_clf == 0
        stripe_w = int(x.size(2) // self.num_clf)
        local_feat_list = []
        local_preds_list = []
        for i in range(self.num_clf):
            local_feat = x[:, :, :, i * stripe_w: (i + 1) * stripe_w]
            local_feat_list.append(self.local_FE_list[i](local_feat))
            local_preds_list.append(self.local_clf_list[i](local_feat_list[i]))
        final_clf = self.total_clf(torch.cat(local_feat_list,dim=1).detach())
        local_preds_list.append(final_clf)
        return local_preds_list,local_feat_list
