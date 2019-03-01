from fastprogress import master_bar, progress_bar
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd
from torch import optim
import re
import torch
from fastai import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import pretrainedmodels
from collections import OrderedDict
import math
from arch import *
from utils import *
from losses import *
import torchvision
df = pd.read_csv('data/train.csv')
val_fns = pd.read_pickle('data/val_fns')

fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)

SZ = 384
BS = 20
NUM_WORKERS = 10
SEED=0
SAVE_TRAIN_FEATS = False
SAVE_TEST_MATRIX = False
LOAD_IF_CAN = True

name = f'VGG16-GeMConst-bbox-PCB4-{SZ}-val-Ring-CELU'

data = (
    ImageItemListGray
        .from_df(df[df.Id != 'new_whale'], 'data/crop_train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('data/crop_test'))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)
class CustomPCBNetwork(nn.Module):
    def __init__(self, new_model):
        super().__init__()
        self.cnn =  new_model.features
        self.head = PCBRingHead2(5004, 512, 4, 512)
    def forward(self, x):
        x = self.cnn(x)
        out = self.head(x)
        return out

learn = Learner(data, CustomPCBNetwork(torchvision.models.vgg16_bn(pretrained=True)),
                   metrics=[map5ave,map5total],
                   loss_func=MultiCE,
                   callback_fns = [RingLoss])
learn.split([learn.model.cnn[26], learn.model.head])
learn.freeze()
learn.clip_grad();
LOADED = False
print ("Stage one, training only head")
if LOAD_IF_CAN:
    try:
        learn.load(name)
        LOADED = True
    except:
        LOADED = False
if not LOADED:
    learn.fit_one_cycle(100, 1e-2/1.5)#6.7e-3
    learn.save(name)
print ('Stage 1 done, finetuning everything')

learn.unfreeze()
max_lr = 2e-3
lrs = [max_lr/10., max_lr, max_lr]

LOADED = False
if LOAD_IF_CAN:
    try:
        learn.load(name+ '_unfreeze')
        LOADED = True
    except:
        LOADED = False

if not LOADED:
    learn.fit_one_cycle(100, lrs)
    learn.save(name + '_unfreeze')
print ("Stage 2 done, starting stage 3")

LOADED = False
print ("Stage 2 done, stage 3 done")


####### Validation
print ("Starting validation")
df = pd.read_csv('data/train.csv')
val_fns = pd.read_pickle('data/val_fns')
new_whale_fns = set(df[df.Id == 'new_whale'].sample(frac=1).Image.iloc[:1000])
y = val_fns.union(new_whale_fns)
classes = learn.data.classes + ['new_whale']
data = (
    ImageItemListGray
        .from_df(df, 'data/crop_train', cols=['Image']) 
        .split_by_valid_func(lambda path: path2fn(path) in y)
        .label_from_func(lambda path: fn2label[path2fn(path)], classes=classes)
        .add_test(ImageItemList.from_folder('data/crop_test'))
        .transform(get_transforms(do_flip=False, max_zoom=1,
                                  max_warp=0,
                                  max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)
data.train_dl.dl.batch_sampler.sampler = torch.utils.data.SequentialSampler(data.train_ds)
data.train_dl.dl.batch_sampler.drop_last = False
data.valid_dl.dl.batch_sampler.sampler = torch.utils.data.SequentialSampler(data.valid_ds)
data.valid_dl.dl.batch_sampler.drop_last = False

learn.data = data
targs = torch.tensor([classes.index(label.obj) if label else 5004 for label in learn.data.valid_ds.y])

####
val_preds, val_gt,val_feats,val_preds2 = get_predictions(learn.model,data.valid_dl)
print ("Finding softmax coef")
best_preds, best_th, best_sm_th, best_score = find_softmax_coef(val_preds,targs, [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0])

####### Now features
print ("Extracting train feats")
train_feats, train_labels = get_train_features(learn, augment=0)
distance_matrix_imgs = batched_dmv(val_feats, train_feats)
distance_matrix_classes = dm2cm(distance_matrix_imgs, train_labels)
class_sims = 0.5*(2.0 - distance_matrix_classes)
class_sims_th, best_th_feats, score_feats_th = find_new_whale_th(class_sims, targs)
out_preds, thlist, best_score = find_mixing_proportions(best_preds,
                                                       class_sims,
                                                      class_sims_th,targs)
out_preds = out_preds.cuda()
targs = targs.cuda()
print ("Best mix score = ", best_score)
print ("Val top1 acc = ", accuracy(out_preds, targs).cpu().item())
print ("Val map5 = ",map5(out_preds, targs).cpu().item())
print ("Val top5 acc = ",top5acc(out_preds, targs).cpu().item())
thresholds = {}
thresholds['softmax'] = best_sm_th
thresholds['preds_th'] = best_th
thresholds['preds_th_feats'] = best_th_feats
thresholds['mix_list'] = thlist
torch.save(thresholds, name + '_thresholds.pth')

if SAVE_TRAIN_FEATS:
    print ("Saving train feats")
    torch.save({"train_labels": train_labels.detach().cpu(), 
                "train_feats": train_feats.detach().cpu(),
                "val_labels": targs,
                "val_feats": val_feats.detach().cpu(),
                "classes": classes,
                "thresholds": thresholds,
                }, name + 'train_val_feats.pt')



###############
#Test
test_preds,  test_gt,test_feats,test_preds2 = get_predictions(learn.model,data.test_dl)
preds_t = torch.softmax(best_sm_th * test_preds, dim=1)
preds_t = torch.cat((preds_t, torch.ones_like(preds_t[:, :1])), 1)
preds_t[:, 5004] = best_th
#Concat with val
all_gt0 = torch.cat([val_gt, train_labels], dim=0)
all_feats0 = torch.cat([val_feats, train_feats], dim=0)
dm3 = batched_dmv(test_feats, all_feats0)
cm3 = dm2cm(dm3, all_gt0)
cm3 = 0.5*(2.0 - cm3)
preds_ft_0t = cm3.clone().detach()
preds_ft_0t[:, 5004] = best_th_feats
pit1 = thlist[0]*cm3 + thlist[1]*preds_ft_0t+thlist[2]*preds_t
if SAVE_TEST_MATRIX:
    print ("Saving test feats")
    torch.save({"test_feats": test_feats.detach().cpu(), 
                "best_preds": pit1.detach().cpu(),
                "classes": classes,
                "thresholds": thresholds,
                }, name + 'test_feats.pt')    
try:
    os.makedirs('subs')
except:
    pass
create_submission(pit1.cpu(), learn.data, name, classes)
print ('new_whales at 1st pos:', pd.read_csv(f'subs/{name}.csv.gz').Id.str.split().apply(lambda x: x[0] == 'new_whale').mean())
