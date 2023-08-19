from torch.utils.data import Dataset, DataLoader
import glob
import os
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler, RandomSampler
from ever.api.data import CrossValSamplerGenerator
import numpy as np
import cv2 as cv
import logging
import torch
import pickle
import random
from utils.tools import seed_worker
from PIL import Image
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)



# LABEL_MAP = OrderedDict(
#     Water=1,
#     Artificial_Bareground=2,
#     Natural_Bareground=3,
#     Permanent_Snow=4,
#     Woody_Vegetation=5,
#     Cultivated_Vegetation=6,
#     Semi_Natural_Vegetation=7
# )

LABEL_MAP = OrderedDict(
    Background=1,
    Building=2,
    Road=3,
    Water=4,
    Barren=5,
    Forest=6,
    Agricultural=7
)

# LABEL_MAP = OrderedDict(
#     No_data=0,
#     Cultivated_land=1,
#     Forest=2,
#     Grassland=3,
#     Shrubland=4,
#     Water=5,
#     Artificial_urface=6,
#     Bareland=7
# )




class LoveDA(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, edge_channel=False, training=True):
        self.rgb_filepath_list = []
        self.cls_filepath_list = []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)

        else:
            self.batch_generate(image_dir, mask_dir)

        X = np.array(self.rgb_filepath_list)
        y = np.array(self.cls_filepath_list)
        X, y = shuffle(X, y, random_state=0)

        self.rgb_filepath_list = list(X)
        self.cls_filepath_list = list(y)

        self.transforms = transforms
        self.image_dir = image_dir
        self.edge_channel = edge_channel
        self.training = training

        self.all_urban_names = set(os.listdir('LoveDA_mixedDistricts/Train/Urban512_fine/images_png'))
        self.all_rural_names = set(os.listdir('LoveDA_mixedDistricts/Train/Rural512_fine/images_png'))
        self.all_urban_names = set(os.listdir('processed_slovenia/train/urban/img_finetune'))
        self.all_rural_names = set(os.listdir('processed_slovenia/train/rural/img_finetune'))


        # with open('saved_dict_urban512.pkl', 'rb') as f1:
        #     self.urban_dict = pickle.load(f1)
        # with open('saved_dict_rural512.pkl', 'rb') as f2:
        #     self.rural_dict = pickle.load(f2)
    


    def transfer_color(self, image, mask, image_name):
        # print(image.shape)
        # print(mask.shape)
        mask = torch.tensor(mask)

        if (image_name in self.all_urban_names):
            if(image_name not in self.urban_dict.keys()):
                return image
            while(True):
                key = random.sample(list(self.rural_dict), 1)[0]
                if (key in self.all_rural_names):
                    diff = np.array(self.rural_dict[key]) - np.array(self.urban_dict[image_name])
                    chosen_key = self.rural_dict[key]
                    break
        else:
            while(True):
                if(image_name not in self.rural_dict.keys()):
                    return image
                key = random.sample(list(self.urban_dict), 1)[0]
                if (key in self.all_urban_names):
                    diff = np.array(self.urban_dict[key]) - np.array(self.rural_dict[image_name])
                    chosen_key = self.urban_dict[key]
                    break

        for cc in range(1, 7):
            if (chosen_key[0][cc] == -1):
                continue
            operator_0 = torch.where(mask==cc, diff[0][cc], 0.)
            operator_1 = torch.where(mask==cc, diff[1][cc], 0.)
            operator_2 = torch.where(mask==cc, diff[2][cc], 0.)

            operator_0 = operator_0.cpu().detach().numpy()
            operator_1 = operator_1.cpu().detach().numpy()
            operator_2 = operator_2.cpu().detach().numpy()

            image[:, :, 0] = np.clip(image[:, :, 0]+operator_0, a_min=0, a_max=255)
            image[:, :, 1] = np.clip(image[:, :, 1]+operator_1, a_min=0, a_max=255)
            image[:, :, 2] = np.clip(image[:, :, 2]+operator_2, a_min=0, a_max=255)


        # im = Image.fromarray(image)
        # im.save("real_time_output/"+image_name)
        return image


    def batch_generate(self, image_dir, mask_dir):
        # Change here !!!! 
        #rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.npy'))
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list



    def __getitem__(self, idx):
        # Change here !!!! 
        #image = np.load(self.rgb_filepath_list[idx])
        image = imread(self.rgb_filepath_list[idx])
        
        #image_id = self.rgb_filepath_list[idx].split("/")[-1][:-4]
        #image_name = self.rgb_filepath_list[idx].split("/")[-1]


        if len(self.cls_filepath_list) > 0:
            
            # edges = cv.Canny(image, 100, 200)
            # Change here !!!! 
            # if(self.edge_channel):
            #     edges = np.load(self.image_dir[-1]+image_id+'.npy')
            #     edges = np.expand_dims(edges, axis=2)
            #     image = np.concatenate((image, edges), axis=2)

            # Change here !!!! 
            #mask = np.squeeze(np.load(self.cls_filepath_list[idx][:-9]+'mask_'+self.cls_filepath_list[idx][-5:])).astype(np.long)
            mask = imread(self.cls_filepath_list[idx]).astype(np.long) -1
            mask = np.where(mask == 4, 0, mask)
            mask = np.where(mask > 4, mask-1, mask)

            # if(random.random() < 0.3 and self.training):
            # # if (True):
            #     image = self.transfer_color(image, mask, image_name)
            
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']

            return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']

            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)



class LoveDALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = LoveDA(self.config.image_dir, self.config.mask_dir, self.config.transforms, self.config.edge_channel, self.config.training)

        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(LoveDALoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True,
                                       drop_last=True
                                       )
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            scale_size=None,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))
