"""
This file generates the 2D binary masks for each patient using the TIFF data.
"""
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from tqdm import tqdm
import sys
sys.path.append("/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Code/Automated Tumor Segmentation/")
from utils import create_mask, erosion
from config import parse_args

args = parse_args()
path_TIFF = args.path_TIFF_data
for i in tqdm(sorted(os.listdir(path_TIFF))):
    try:
        ID = i + args.annotations
        path = os.path.join(path_TIFF, i, ID)
        mask_dir = os.path.join(path_TIFF, i, 'Mask_' + args.annotations)
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)

        for j in sorted(os.listdir(path)):
            # Create Mask
            mask = create_mask(os.path.join(path, j), addInterior=True)
            # Save Mask
            mask = Image.fromarray(mask)
            save_path = os.path.join(mask_dir, j.split('.')[0] + '.png')
            mask.save(save_path)
    except:
        print("Masks cannot be generated for the Patient: ", i)
print("Done!")