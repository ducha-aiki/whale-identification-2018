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
from arch import *
def map5(preds,targs):
    if type(preds) is list:
        return torch.cat([map5fast(p, targs).view(1) for p in preds ]).mean()
    return map5fast(preds,targs)
def map5fast(preds, targs, k =5):
    top_5 = preds.topk(k, 1)[1]
    targs = targs.to(preds.device)
    scores = torch.zeros(len(preds), k).float().to(preds.device)
    for kk in range(k):
        scores[:,kk] = (top_5[:,kk] == targs).float() / float(kk+1)
    return scores.max(dim=1)[0].mean()

def map5sigm(preds, targs):
    targs = torch.max(targs,dim=1)[1]
    predicted_idxs = preds.sort(descending=True)[1]
    top_5 = predicted_idxs[:, :5]
    res = mapk([[t] for t in targs.cpu().numpy()], top_5.cpu().numpy(), 5)
    return torch.tensor(res)


def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]

def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels

def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')

    
def find_new_whale_th(preds, targs):
    with torch.no_grad():
        if preds.size(1) == 5004:
            preds2 = torch.cat((preds, torch.ones_like(preds[:, :1])), 1)
        else:
            preds2 = preds.clone()
        res = []
        ps = np.linspace(0, 0.95, 121)
        for p in ps:
            preds2[:, 5004] = p
            res.append(map5(preds2, targs).item())
        best_p = ps[np.argmax(res)]
        #print (res, best_p)
        preds2[:, 5004] = best_p
        score = map5(preds2, targs)
        print('score=',score, 'th=',best_p)
    return preds2, best_p, score
def find_softmax_coef(preds, targs, softmax_coefs = [0.5, 0.75, 1.0, 1.5, 2.0, 2.2]):
    best_preds = None
    best_th = -100
    best_score = 0
    best_sm_th = 0
    for sc in softmax_coefs:
        sm_preds = torch.softmax(sc*preds, dim=1).cpu()
        preds2, best_p, score = find_new_whale_th(sm_preds, targs)
        if score > best_score:
            best_preds = preds2
            best_th = best_p
            best_score = score
            best_sm_th = sc
    print ('best softmax=', best_sm_th)
    return best_preds, best_th, best_sm_th, best_score
def get_train_features(learn, augment=3):
    ####### Now features
    torch.cuda.empty_cache()
    try:
        all_preds0, all_gt0,all_feats0,all_preds20 = get_predictions(learn.model,learn.data.train_dl)
    except:
        all_preds0, all_gt0,all_feats0,all_preds20 = get_predictions_non_PCB(learn.model,learn.data.train_dl)
    for i in range(max(augment,0)):
        torch.cuda.empty_cache()
        try:
            all_preds00, all_gt00,all_feats00,all_preds200 = get_predictions(learn.model,learn.data.train_dl)
        except:
            all_preds00, all_gt00,all_feats00,all_preds200 =  get_predictions_non_PCB(learn.model,learn.data.train_dl)
        all_gt0 = torch.cat([all_gt0, all_gt00], dim=0)
        all_feats0 = torch.cat([all_feats0, all_feats00], dim=0)
    train_feats = all_feats0
    train_labels = all_gt0
    return train_feats, train_labels
def find_mixing_proportions(sm_preds, sim, sim_th, targs):
    best_score = 0
    out_preds = None
    for c1 in range(5):
        for c2 in range(5):
            for c3 in range(5):
                c31 = float(c3)*0.2
                out_with_feats = c1*sim + c2*sim_th+c31*sm_preds
                score = map5(out_with_feats, targs)
                if score > best_score:
                    best_score= score
                    thlist = [c1,c2,c31]
                    out_preds = out_with_feats
                    
    return out_preds, thlist, best_score

def get_predictions(model, val_loader):
    torch.cuda.empty_cache()
    model.eval()
    all_preds = []
    all_confs = []
    all_feats= []
    all_preds2 = []
    all_gt = []
    c= 0
    with torch.no_grad():
        for data1,label in val_loader:
            preds_list,feats_list = model(data1)
            all_preds.append(preds_list[-1].cpu())
            all_preds2.append(torch.stack(preds_list[:-1],-1).cpu())
            all_gt.append(label.cpu())
            all_feats.append(L2Norm()(torch.cat(feats_list, dim=1)).cpu())
            #all_confs.append(confs)
        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_feats = torch.cat(all_feats, dim=0).cpu()
        #all_confs = torch.cat(all_confs, dim=0)
        pred_clc = all_preds.max(dim=1)[1].cpu()
        all_gt = torch.cat(all_gt, dim=0).cpu()
        mp5 = map5(all_preds,all_gt).mean()
        acc = (pred_clc==all_gt).float().mean().detach().cpu().item()
        out = f'acc = {acc:.3f}, map5 = {mp5:.3f}'
        print(out)
    return all_preds, all_gt,all_feats,all_preds2
def get_predictions_non_PCB(model, val_loader):
    torch.cuda.empty_cache()
    model.eval()
    all_preds = []
    all_confs = []
    all_feats= []
    all_preds2 = []
    all_gt = []
    c= 0
    with torch.no_grad():
        for data1,label in val_loader:
            preds,feats, feats2 = model(data1)
            all_preds.append(preds.cpu())
            all_feats.append(torch.cat([L2Norm()(feats).cpu(),L2Norm()(feats2).cpu()], dim=1))
            all_gt.append(label.cpu())
            #all_confs.append(confs)
        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_feats = torch.cat(all_feats, dim=0).cpu()
        pred_clc = all_preds.max(dim=1)[1].cpu()
        all_gt = torch.cat(all_gt, dim=0).cpu()
        mp5 = map5(all_preds,all_gt).mean()
        acc = (pred_clc==all_gt).float().mean().detach().cpu().item()
        out = f'acc = {acc:.3f}, map5 = {mp5:.3f}'
        print(out)
    return all_preds, all_gt,all_feats,all_preds2
def distance_matrix_vector(anchor, positive, d2_sq):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)
def dm2cm(dm, labels):
    cl = set(labels.detach().cpu().numpy())
    n_cl = len(cl)
    dists = torch.zeros(dm.size(0),n_cl)
    for i in range(n_cl):
        mask = labels == i
        dists[:,i] = dm[:,mask].min(dim=1)[0]
    return dists

def dm2cm_with_idxs(dm, labels):
    cl = set(labels.detach().cpu().numpy())
    n_cl = len(cl)
    dists = torch.zeros(dm.size(0),n_cl)
    idxs = torch.zeros(dm.size(0),n_cl)
    for i in range(n_cl):
        mask = labels == i
        tt = dm[:,mask].min(dim=1)
        dists[:,i] = tt[0]
        iiiii = torch.arange(dm.size(1)).unsqueeze(0).expand_as(dm)[:,mask]
        for j in range(len(tt[1])):
            idxs[j,i] = iiiii[0,tt[1][j]]
    return dists, idxs

def get_train_val_fnames(df, val_list):
    train_fnames = []
    val_fnames = []
    for i in df.Image:
        if i not in val_list:
            train_fnames.append(str(i))
        else:
            val_fnames.append(str(i))
    return train_fnames, val_fnames

def get_shortlist_fnames(distance_matrix_idxs, class_sims, df, val_list):
    train_fnames, val_fnames = get_train_val_fnames(df, val_list)
    best_scores, best_idxs = torch.topk(class_sims,5, 1)
    shortlist_dict = {}
    for i, fname in enumerate(val_fnames):
        bi = best_idxs[i]
        ci = distance_matrix_idxs[i][bi]
        sl = []
        for iii in ci:
            sl.append(train_fnames[int(iii)])
        shortlist_dict[fname] = sl
    return shortlist_dict
def get_shortlist_fnames_test(distance_matrix_idxs, class_sims, df, learn, val_list ):
    train_fnames, val_fnames = get_train_val_fnames(df, val_list)
    train_fnames = val_fnames + train_fnames
    test_fnames = []
    for path in learn.data.test_ds.x.items:
        test_fnames.append(path.name)
    best_scores, best_idxs = torch.topk(class_sims,5, 1)
    shortlist_dict = {}
    for i, fname in enumerate(test_fnames):
        bi = best_idxs[i]
        ci = distance_matrix_idxs[i][bi]
        sl = []
        for iii in ci:
            sl.append(train_fnames[int(iii)])
        shortlist_dict[fname] = sl
    return shortlist_dict


def batched_dmv(d1,d2):
    torch.cuda.empty_cache()
    out = torch.zeros(d1.size(0), d2.size(0))
    d2_sq1 = torch.sum(d2**2, dim=1).unsqueeze(-1)
    try:
        out = distance_matrix_vector(d1.cuda(),d2.cuda(),d2_sq1.cuda()).cpu()
    except:
        out = distance_matrix_vector(d1,d2,d2_sq1).cpu()
    return out

def open_image_grey(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image)->Image:
    "Return `Image` object created from image in file `fn`."
    #fn = getattr(fn, 'path', fn)
    x = PIL.Image.open(fn).convert(convert_mode).convert('LA').convert(convert_mode)
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(x)



class ImageItemListGray(ImageItemList):
    def open(self, fn:PathOrStr)->Image:
        return open_image_grey(fn)
    
def top5acc(preds, targs):
    predicted_idxs = preds.sort(descending=True)[1]
    top_5 = predicted_idxs[:, :5]
    res = (targs.unsqueeze(1).expand_as(top_5) == top_5).float().max(dim=1)[0].mean()
    return res
def map5ave(preds, targs):
    pl = len(preds)
    out = torch.stack(preds[:pl-1], -1).mean(dim=-1)
    return map5(out,targs)
def map5total(preds, targs):
    out = preds[-1]
    return map5(out, targs)
