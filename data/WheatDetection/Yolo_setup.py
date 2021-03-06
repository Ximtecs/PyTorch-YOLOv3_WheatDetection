import pandas as pd
import numpy as np
import re
from PIL import Image
import PIL
import os
import albumentations as A
import albumentations
#from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox
import cv2

train_df = pd.read_csv('train.csv')
#train_df['source'] = 0 #Competition is only about detection - so labels does not matter
#TODO ['source] should be set to 0 after augmentations

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
train_df['x'] = train_df['x'].astype(np.float32)
train_df['y'] = train_df['y'].astype(np.float32)
train_df['w'] = train_df['w'].astype(np.float32)
train_df['h'] = train_df['h'].astype(np.float32)


train_df['area'] = train_df['w'] * train_df['h']
#remove large and small boxes
train_df = train_df[train_df['area'] < 100000] # large BBoxes
train_df = train_df[(train_df['area'] <= 0.0) | (train_df['area'] > 14.0)] # small BBoxes

#translate to center coordinates
train_df['x'] += train_df['w'] / 2.0
train_df['y'] += train_df['h'] / 2.0

#scale according to image width/height
train_df['x'] /= train_df['width']
train_df['y'] /= train_df['height']
train_df['w'] /= train_df['width']
train_df['h'] /= train_df['height']



#------------------------------------ AUGMENTATIONS -----------------

#number of images per class
class_names =  train_df['source'].unique()
p = train_df.groupby('source')['image_id'].nunique()
tot_imgs = sum(p)

img_per_class = 1250

augmentation = A.Compose([      
        A.Resize(416, 416),    
        A.VerticalFlip(p=0.60),     # Verticlly flip the image
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.60), # brightness
        A.HueSaturationValue(p=0.60, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50) # HUE
    ], bbox_params={'format': 'albumentations', 'label_fields': ['labels']})

def augment_img(image_id, i):

    img = cv2.imread(f'images/{image_id}.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #TODO - x_max and y_max should be changed 
    bboxes = train_df[train_df['image_id'] == image_id][['x', 'y', 'w', 'h']].astype(np.float32).values
    bboxes[:,0] = np.minimum(np.maximum(bboxes[:,0] - bboxes[:,2] / 2.0,0.0),1.0) # x center to xmin
    bboxes[:,1] = np.minimum(np.maximum(bboxes[:,1] - bboxes[:,3] / 2.0,0.0),1.0) # y center to ymax
    bboxes[:,2] = np.minimum(np.maximum(bboxes[:,0] + bboxes[:,2],0.0),1.0)     #width to xmax
    bboxes[:,3] = np.minimum(np.maximum(bboxes[:,1] + bboxes[:,3],0.0),1.0)     #height to ymax
    source = train_df[train_df['image_id'] == image_id]['source'].unique()[0]
    labels = np.ones((len(bboxes), ))  # as we have only one class (wheat heads)
    aug_result = augmentation(image=img, bboxes=bboxes, labels=labels)
    aug_image = Image.fromarray(aug_result['image'])
    aug_bboxes = np.array(aug_result['bboxes'])

    aug_bboxes[:,2] = np.minimum(np.maximum(aug_bboxes[:,2] - aug_bboxes[:,0],0.0),1.0)     #xmax to width
    aug_bboxes[:,3] = np.minimum(np.maximum(aug_bboxes[:,3] - aug_bboxes[:,1],0.0),1.0)     #ymax to height
    aug_bboxes[:,0] = np.minimum(np.maximum(aug_bboxes[:,0] + aug_bboxes[:,2] / 2.0,0.0),1.0) # xmin to center x
    aug_bboxes[:,1] = np.minimum(np.maximum(aug_bboxes[:,1] + aug_bboxes[:,3] / 2.0,0.0),1.0) # ymin to center y

    #Save augmented image
    aug_image.save(f'images/{image_id}_aug_{i}.jpg')

    image_metadata = []
    for bbox in aug_bboxes:
        image_metadata.append({
            'image_id': f'{image_id}_aug_{i}',
            'width': 416, #416
            'height': 416, #416
            'x': bbox[0],
            'y': bbox[1],
            'w': bbox[2],
            'h': bbox[3] ,
            'area': bbox[2] * bbox[3],
            'source': source
        })
    return image_metadata

augmented_imgs = list()
augment_df = pd.DataFrame(augmented_imgs)
for name in class_names:
    print(f"augmenting {name}")
    class_ids = train_df.groupby('source').get_group(name)
    no_class_imgs  = p[name]

    if no_class_imgs > 600:
        no_aug = img_per_class - no_class_imgs
        df_class_idx = class_ids.sample(n=no_aug, replace=True)
        img_ids = image_ids = df_class_idx['image_id'].unique()
    else:
        img_ids = class_ids['image_id'].unique()
    nr_iterations = img_per_class // no_class_imgs
    i = 0
    while i < nr_iterations:
        for img_id in img_ids:
            augmented_imgs.extend(augment_img(img_id,i))
        i += 1

augment_df = pd.DataFrame(augmented_imgs)
#--------------------------------------------------------------------


#crop images to 416x416 size - expected input of YOLO
img_ids = train_df['image_id'].unique()
for id_ in img_ids:
    img = Image.open(f'images/{id_}.jpg')
    img = img.resize((416,416),Image.ANTIALIAS)
    img.save(f'images/{id_}.jpg')
#--------------------------------------------------

#------ COMBINE TRAINING AND AUGMENTATION DATAFRAMES --------------
train_df = pd.concat([train_df,augment_df],axis=0)
#---------------------------------------------

# -----------pick training and validation set randomly -----------------------
img_ids = train_df['image_id'].unique()
choice = np.random.choice(range(img_ids.shape[0]), size=(1000,), replace=False)    
ind = np.zeros(img_ids.shape[0], dtype=bool)
ind[choice] = True
rest = ~ind
training_ids = img_ids[rest]
valid_ids = img_ids[ind]
train_red_df = train_df[train_df['image_id'].isin(training_ids)]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
#--------------------------------------------------------------

# -----Create list of training images--------------
path = "data/WheatDetection/images/"
file = open('train.txt','w')
for id_ in training_ids:
    name = f'{path}{str(id_)}.jpg\n'
    file.write(name)
file.close()
#----------------------------------------------

# ------create list of validation images -------
file = open('valid.txt','w')
for id_ in valid_ids:
    name = f'{path}{str(id_)}.jpg\n'
    file.write(name)
file.close()
#-----------------------------------------------------------


# -------- Create label files for all images ----------
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

# ----------- resize test images
test_imgs = os.listdir("test/")
for img_id in test_imgs:
    img = Image.open(f'test/{img_id}')
    img = img.resize((416,416),Image.ANTIALIAS)
    img.save(f'test/{img_id}')