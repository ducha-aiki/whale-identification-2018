import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
from time import time
df = pd.read_csv('data/train.csv')
df.head()
bbox_df = pd.read_csv('data/bounding_boxes.csv')
bbox_df.head()
df = df.merge(bbox_df, on=['Image'])
def crop_loose_bbox(img,area, val=0.2):
    img_w, img_h = img.size
    w = area[2] - area[0]
    h = area[3] - area[1]
    area2 = (max(0, int(area[0] - 0.5*val*w)),
             max(0, int(area[1] - 0.5*val*h)),
             min(img_w, int(area[2] + 0.5*val*w)),
             min(img_h, int(area[3] + 0.5*val*h)))
    return img.crop(area2)
try:
    os.makedirs('data/crop_train')
    os.makedirs('data/crop_test')
except:
    pass
    
t=time()
for i in range(len(df)):
    fn = os.path.join('data/train', df.Image[i])
    img = Image.open(fn)
    area = (df.x0[i],df.y0[i],df.x1[i],df.y1[i])
    cropped_img = crop_loose_bbox(img,area, 0.2)
    cropped_img.save(fn.replace('train', 'crop_train'))
    if i %1000 == 0:
        print(i)
        print(time() -t , 'sec')
print(time() -t )
df = pd.read_csv('data/sample_submission.csv')
df.head()
bbox_df = pd.read_csv('data/bounding_boxes.csv')
bbox_df.head()
df = df.merge(bbox_df, on=['Image'])

t=time()
for i in range(len(df)):
    fn = os.path.join('data/test', df.Image[i])
    img = Image.open(fn)
    area = (df.x0[i],df.y0[i],df.x1[i],df.y1[i])
    cropped_img = crop_loose_bbox(img,area, 0.2)
    cropped_img.save(fn.replace('test', 'crop_test'))
    if i %1000 == 0:
        print(i)
        print(time() -t , 'sec')
print(time() -t )
