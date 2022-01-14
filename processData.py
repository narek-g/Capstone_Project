# proprocess data 

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json 

import os



class ProcessData(object):
    
    def __init__(self, df, Height, Width):
        self.HEIGHT = Height
        self.WIDTH = Width
        self.category_num = 46+1 
        self.df = df
        
    
    def make_mask_img(self, segment_df):
        seg_width = segment_df.at[0, "Width"]
        seg_height = segment_df.at[0, "Height"]
        seg_img = np.full(seg_width*seg_height, self.category_num-1, dtype=np.int32)
        for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
            pixel_list = list(map(int, encoded_pixels.split(" ")))
            for i in range(0, len(pixel_list), 2):
                start_index = pixel_list[i] - 1
                index_len = pixel_list[i+1] - 1
                seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
        seg_img = seg_img.reshape((seg_height, seg_width), order='F')
        seg_img = cv2.resize(seg_img, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
        return seg_img 
    
    
    def get_image_batch(self): 
        data_dir = '/Users/narekgeghamyan/local_data/capstone_data/imaterialist-fashion-2019-FGVC6/train/'
        # data_list = os.listdir('/Users/narekgeghamyan/local_data/capstone_data/imaterialist-fashion-2019-FGVC6/train/')
        img_ind_num = self.df.groupby("ImageId")["ClassId"].count()
        index = self.df.index.values[0]
        trn_images = []
        seg_images = []
        img_names = []
        for i, (img_name, ind_num) in enumerate(img_ind_num.items()):
            img = cv2.imread(data_dir + img_name)
            img = cv2.resize(img, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)
            segment_df = (self.df.loc[index:index+ind_num-1, :]).reset_index(drop=True)
            index += ind_num
            if segment_df["ImageId"].nunique() != 1:
                raise Exception("Index Range Error")
            seg_img = self.make_mask_img(segment_df)
            
            # HWC -> CHW
            # img = img.transpose((2, 0, 1))
            #seg_img = seg_img.transpose((2, 0, 1))
            
            trn_images.append(np.array(img, dtype=np.float32)/255 )
            seg_images.append(np.array(seg_img, dtype=np.int32))
            img_names.append(img_name)
        return(trn_images, seg_images, img_names)