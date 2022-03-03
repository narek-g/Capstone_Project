# python class which computes and returns image segmentation mask 

import os 
from PIL import Image
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict

from base_dataset import Normalize_image 

from model import U2NET

class ImagePredictor(): 
    def __init__(self):
        self.checkpoint_path = "/Users/narekgeghamyan/Classes/MLE_bootcamp/Capstone_Project/Production/Deployment/model/u2net.pth"
        self.result_dir     = "/Users/narekgeghamyan/Classes/MLE_bootcamp/Capstone_Project/Production/Deployment/Results"
        self.net = U2NET(in_ch=3, out_ch=4)
        self.num_classes = 4
        
    def get_palette(self):
        """Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = self.num_classes
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
                palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
                palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
                i += 1
                lab >>= 3
        return palette
    
    def load_checkpoint_mgpu(self):
        if not os.path.exists(self.checkpoint_path):
            print("----No checkpoints at given path----")
            return
        model_state_dict = torch.load(self.checkpoint_path, map_location=torch.device("cpu"))
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        self.net.load_state_dict(new_state_dict)
        return(self.net)

    
    def make_prediction(self, image): 
        self.img = image
        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(0.5, 0.5)]
        transform_rgb = transforms.Compose(transforms_list)
        
        image_name = 'segmented_image.jpg'

        palette = self.get_palette()
        self.net = self.load_checkpoint_mgpu()
        
        image_tensor = transform_rgb(self.img)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        output_tensor = self.net(image_tensor) 
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

        output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
        output_img.putpalette(palette)
        output_img.save(os.path.join(self.result_dir, image_name[:-3] + "png"))
