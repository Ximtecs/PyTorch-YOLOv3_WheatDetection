import pandas as pd
import numpy as np
import re
from PIL import Image
import PIL


train_df = pd.read_csv('train.csv')
train_df['source'] = 0 #Competition is only about detection - so labels does not matter


#expand the bbox to 4 different columns
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1
def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r
train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)



#translate to center coordinates
train_df['x'] += + train_df['w'] / 2
train_df['y'] += + train_df['h'] / 2

#scale according to image width/height
train_df['x'] /= train_df['width']
train_df['y'] /= train_df['height']
train_df['w'] /= train_df['width']
train_df['h'] /= train_df['height']


#pick training and validation set randomly
img_ids = train_df['image_id'].unique()

choice = np.random.choice(range(img_ids.shape[0]), size=(655,), replace=False)    
ind = np.zeros(img_ids.shape[0], dtype=bool)
ind[choice] = True
rest = ~ind

training_ids = img_ids[rest]
valid_ids = img_ids[ind]

train_red_df = train_df[train_df['image_id'].isin(training_ids)]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]

#Create list of training images
path = "data/WheatDetection/images/"
file = open('train.txt','w')
for id_ in training_ids:
    name = f'{path}{str(id_)}.jpg\n'
    file.write(name)
file.close()

#create list of validation images
file = open('valid.txt','w')
for id_ in valid_ids:
    name = f'{path}{str(id_)}.jpg\n'
    file.write(name)
file.close()

#Create label files for all images
for id_ in img_ids:
    file = open(f'labels/{id_}.txt','w')
    boxes = (train_df[train_df['image_id'] == id_])[['x','y','w','h']].values
    for i in range(len(boxes)):
        string = ""
        string += str(0) + " " #just detect - no classification
        string += str(boxes[i,0])+ " "
        string += str(boxes[i,1])+ " "
        string += str(boxes[i,2])+ " "
        string += str(boxes[i,3])
        string += "\n"
        file.write(string)
    file.close()

#crop images to 416x416 size - expected input of YOLO
for id_ in img_ids:
    img = Image.open(f'images/{id_}.jpg')
    img = img.resize((416,416),Image.ANTIALIAS)
    img.save(f'images/{id_}.jpg')