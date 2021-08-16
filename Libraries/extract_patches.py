import sys
sys.path.append("/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Code/Automated Tumor Segmentation/")
sys.path.append("/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Code/Automated Tumor Segmentation/Libraries")
import numpy as np
import pandas as pd
from pre_process import my_PreProc, my_PreProc_new
import SimpleITK as sitk
import cv2
import os
from utils import create_mask

def load_Masks(path):
    mask_list = []
    for i in sorted(os.listdir(path)):
        try:
            if i != 'Thumbs.db':
                mask = create_mask(os.path.join(path, i), addInterior=True)
                mask_list.append(mask)
        except:
            print("Mask cannot be generated for the following: ", i)
    mask_arr = np.stack(mask_list, axis=0)     
    return mask_arr

def load_DICOM(path, array=True):
    try:
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_IDs[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3D_dcm = series_reader.Execute()
        if array == True:
            image3D_arr = sitk.GetArrayFromImage(image3D_dcm)
            return image3D_arr
        else:
            return image3D_dcm
    except:
        print("DICOM cannot be loaded with this function. Try others")

def load_csv(path):
    df = pd.read_csv(path)
    df = df.drop(columns='Unnamed: 0')
    return df

def crop_arr(image, mask, args):
#     h = int(image.shape[1]/4.5)
#     w = int(image.shape[2]/4.5)
    h = int(image.shape[1]/args.crop_h)
    w = int(image.shape[2]/args.crop_w)
    image_crop = image[:, h:image.shape[1]-h, w:image.shape[2]-w]
    mask_crop = mask[:, h:image.shape[1]-h, w:image.shape[2]-w]
    return image_crop, mask_crop

def paint_border_overlap(image, patch_h, patch_w, stride_h, stride_w):
    '''
    image & mask are of dimension (#Slice, H, W)
    '''
    assert (len(image.shape)==4)  #4D arrays
    img_h = image.shape[2]  #height of the image
    img_w = image.shape[3] #width of the image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    
    if (leftover_h != 0):  #change dimension of img_h
        tmp_full_imgs = np.zeros((image.shape[0], image.shape[1], img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:image.shape[0], 0:image.shape[1], 0:img_h,0:img_w] = image
        image = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_full_imgs = np.zeros((image.shape[0], image.shape[1], image.shape[2], img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:image.shape[0], 0:image.shape[1], 0:image.shape[2],0:img_w] = image
        image = tmp_full_imgs
    #print("new padded images shape: " +str(full_imgs.shape))
    return image

def extract_ordered_overlap(image, mask, patch_h, patch_w,stride_h,stride_w):
    '''
    Extract image patches in order and overlap
    '''
    assert (len(image.shape)==4 and len(mask.shape)==4)  #4D arrays
    assert (image.shape == mask.shape)
    img_h = image.shape[2]  #height of the full image
    img_w = image.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*image.shape[0]
    
    image_patches = np.empty((N_patches_tot, image.shape[1],patch_h,patch_w))
    mask_patches = np.empty((N_patches_tot, mask.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(image.shape[0]):  #loop over all the slices
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                img_patch = image[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                image_patches[iter_tot] = img_patch
                mask_patch = mask[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                mask_patches[iter_tot] = mask_patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return image_patches, mask_patches  #array with all the full_imgs divided in patches

def load_data(DICOM_path, TIFF_path, args):
    """
    Load the 3D MRI and corresponding Annotation
    """
    image_arr = load_DICOM(DICOM_path)
    mask_arr = load_Masks(TIFF_path)
    
    if image_arr.shape == mask_arr.shape:
        if args.apply_crop == True:
            # Center crop the image and mask
            image_arr_crop, mask_arr_crop = crop_arr(image_arr, mask_arr, args)
        else:
            image_arr_crop, mask_arr_crop = image_arr, mask_arr
        # Preprocessing
#         image_arr_crop, mask_arr_crop = my_PreProc(image_arr_crop, mask_arr_crop)
        image_arr_crop, mask_arr_crop = my_PreProc_new(image_arr_crop, mask_arr_crop)
        
        return image_arr_crop, mask_arr_crop
    else:
        print("DICOM_Path: ", DICOM_path)
        print("TIFF_Path: ", TIFF_path)
        print("Image Arr = {}, Mask Arr = {}".format(image_arr.shape, mask_arr.shape))

# ====================================Load train data==========================================
def get_data_train(df_train_path, patch_height, patch_width, stride_height, stride_width, args):
    df = load_csv(df_train_path)
    # Drop the first row of dataframe
    df = df.drop([df.index[0]]).reset_index(drop=True)
    print("Total number of datasets: ", len(df))
    for index, i in df.iterrows():
        try:
            DICOM_path, TIFF_path = i['DICOM_paths'], i['TIFF_paths']
            # Crop the image and mask
            image_arr_crop, mask_arr_crop = load_data(DICOM_path, TIFF_path, args)
            # Add 0 padding to the image & mask so that it can be divided exactly by the patches dimensions
            image = paint_border_overlap(image_arr_crop, patch_height, patch_width, stride_height, 
                                         stride_width)
            mask = paint_border_overlap(mask_arr_crop, patch_height, patch_width, stride_height, 
                                         stride_width)
            # Extract Patches
            if index == 0:
                image_patches, mask_patches = extract_ordered_overlap(image, mask, patch_height,
                                                                      patch_width, stride_height, 
                                                                      stride_width)
            else:
                image_temp, mask_temp = extract_ordered_overlap(image, mask, patch_height, 
                                                                patch_width, stride_height, stride_width)
                image_patches = np.append(image_patches, image_temp, axis=0)
                mask_patches = np.append(mask_patches, mask_temp, axis=0)
        except:
            print("DICOM and TIFF cannot be loaded for the Pateint ID = {}".format(i['Patient_ID']))
                                       
    return image_patches, mask_patches

# =================================Load test data==========================================
def get_data_test(test_DICOM_path, test_TIFF_path, patch_height, patch_width, stride_height, 
                  stride_width):
    # Load DICOM data
    image_arr = load_DICOM(test_DICOM_path)
    # Load Mask data
    mask_arr = load_Masks(test_TIFF_path)
    # Extract Center Crop
    image_arr_crop = crop_arr_test(image_arr)
    # Preprocessing
    image_arr_crop, mask_arr = my_PreProc(image_arr_crop, mask_arr)
    
    image_list = []
    # Add 0 padding to the image so that it can be divided exactly by the patches dimensions
    image = paint_border_overlap(image_arr_crop, patch_height, patch_width, stride_height, 
                                 stride_width)
    # 'image' is of dimension (#Slices, 1, H, W). After patch extraction, it should be 
    # of dimension (#Slices, #Patches, 1, H, W)
    # Extract Patches
    image_patches = extract_ordered_overlap_test(image, patch_height, patch_width, stride_height,
                                            stride_width)
    
#     return image_patches, image_arr_crop
    return image_patches, image_arr_crop, mask_arr, image.shape[2], image.shape[3]

def extract_ordered_overlap_test(image, patch_h, patch_w,stride_h,stride_w):
#     Extract image patches in order and overlap
    img_h = image.shape[2]  #height of the full image
    img_w = image.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    image_arr = np.empty((image.shape[0], N_patches_img, image.shape[1], patch_h, patch_w))
    for i in range(image.shape[0]):  #loop over all the slices
        iter_tot = 0
        image_temp = np.empty((N_patches_img, image.shape[1], patch_h, patch_w))
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                img_patch = image[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                image_temp[iter_tot, :, :, :] = img_patch
                iter_tot +=1
        image_arr[i] = image_temp
    return image_arr  #array with all the full_imgs divided in patches

def crop_arr_test(image):
    h = int(image.shape[1]/4.5)
    w = int(image.shape[2]/4.5)
#     h = int(image.shape[1]/args.crop_h)
#     w = int(image.shape[2]/args.crop_w)
    image_crop = image[:, h:image.shape[1]-h, w:image.shape[2]-w]
    return image_crop

def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    '''
    recompone the prediction result patches to images
    '''
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k] # Accumulate predicted values
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1  # Accumulate the number of predictions
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0) 
    final_avg = full_prob/full_sum # Take the average
    # print(final_avg.shape)
    assert(np.max(final_avg)<=1.0) # max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) # min value for a pixel is 0.0
    return final_avg

