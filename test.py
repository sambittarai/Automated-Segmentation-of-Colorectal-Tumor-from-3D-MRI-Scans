import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os, sys
import numpy as np
from Libraries.common import setpu_seed, dict_round
from Libraries.logger import Logger, Print_Logger
from Model_Architecture import UNET_Family
from Libraries.metrics import Evaluate
from Libraries.extract_patches import get_data_test, recompone_overlap
from Libraries.dataset import TestDataset
from tqdm import tqdm
from config import parse_args

'''
test_DICOM_path - original image path
test_img_original - original image is converted to array and the center is cropped. Let say (512, 512)
mask_arr - original mask arr without center crop
padded_height & padded_width - padding that is applied to the center cropped image. Let say (524, 524)
'''

class Test():
    def __init__(self, args, test_DICOM_path, test_TIFF_path):
        self.args = args
        self.test_DICOM_path = test_DICOM_path
        self.test_TIFF_path = test_TIFF_path

        #Extract Patches
        self.patches_imgs_test, self.test_img_original, self.mask_arr, self.padded_height, self.padded_width = get_data_test(test_DICOM_path, test_TIFF_path, args.test_patch_height, 
                                               args.test_patch_width, args.stride_height, 
                                               args.stride_width)
        
        self.img_height_original_crop = self.test_img_original.shape[2]
        self.img_width_original_crop = self.test_img_original.shape[3]
        print("Test Set: ", self.patches_imgs_test.shape)
        print("Test_img_original: ", self.test_img_original.shape)

#         test_set = TestDataset(self.patches_imgs_test)
#         self.test_loader = DataLoader(test_set, batch_size=args.batch_size_VS, shuffle=False, 
#                                       num_workers=0)

    def inference(self, net):
        net.eval()
        final_prediction = np.empty((self.test_img_original.shape[0], self.test_img_original.shape[2], self.test_img_original.shape[3]))
        final_prediction_prob = np.empty((self.test_img_original.shape[0], self.test_img_original.shape[2], self.test_img_original.shape[3]))
        
        for i in tqdm(range(self.patches_imgs_test.shape[0])):
            # Create DataLoader
            test_set = TestDataset(self.patches_imgs_test[i])
#             self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
#                                           num_workers=4)
            self.test_loader = DataLoader(test_set, batch_size=4, shuffle=False, 
                                          num_workers=4)
            preds_outputs = []
            preds_prob_dist = []
            with torch.no_grad():
                for batch_idx, inputs in enumerate(self.test_loader, len(self.test_loader)):
                    inputs = inputs.cuda() #if you are using GPU
                    outputs = net(inputs)
                    outputs_prob_dist = outputs[:,1].data.cpu().numpy() #probability distribution
                    outputs_mask = outputs.argmax(dim = 1).data.cpu().numpy() #segmentation mask
                    preds_prob_dist.append(outputs_prob_dist)
                    preds_outputs.append(outputs_mask)
            predictions_mask = np.concatenate(preds_outputs, axis=0)
            predictions_prob_dist = np.concatenate(preds_prob_dist, axis=0)
            self.pred_patches_mask = np.expand_dims(predictions_mask,axis=1)
            self.pred_patches_prob = np.expand_dims(predictions_prob_dist,axis=1)
            
            self.pred_imgs_mask = recompone_overlap(self.pred_patches_mask, self.padded_height, self.padded_width, self.args.stride_height, self.args.stride_width)
            self.pred_imgs_mask = self.pred_imgs_mask[:, :, 0:self.img_height_original_crop, 0:self.img_width_original_crop]
            
            self.pred_imgs_prob_dist = recompone_overlap(self.pred_patches_prob, self.padded_height, self.padded_width, self.args.stride_height, self.args.stride_width)
            self.pred_imgs_prob_dist = self.pred_imgs_prob_dist[:, :, 0:self.img_height_original_crop, 0:self.img_width_original_crop]
            
#             print("Slice Dimension: ", self.pred_imgs_mask.shape)
#             print("Type: ", type(self.pred_imgs_mask.shape))
            final_prediction[i,:,:] = self.pred_imgs_mask[0,0,:,:]
            final_prediction_prob[i,:,:] = self.pred_imgs_prob_dist[0,0,:,:]
        
        return final_prediction, final_prediction_prob

    
if __name__ == '__main__':
    args = parse_args()
    save_path = os.path.join(args.outf, args.save)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    net = UNET_Family.U_Net(args.in_channels, args.classes).to(device)
    cudnn.benchmark = True
    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(os.path.join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])
    test_DICOM_path = "/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Data/Data_DICOM/Pat1_DICOM/Pat1_T2/14LGD1GL/2NJAGREC"
    test_TIFF_path = "/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Data/Data_TIFF/Pat1/Pat1T2M"
    eval = Test(args, test_DICOM_path, test_TIFF_path)
    prediction, prob_distribution = eval.inference(net)
    save_Mask_path = "/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Code/Automated Tumor Segmentation/Experiments/Prediction_Masks/prediction.npy"
    save_prob_path = "/media/sambit/HDD/Sambit/Projects/Radiomics_Project1/Code/Automated Tumor Segmentation/Experiments/Prediction_Masks/prob_distribution.npy"
    np.save(save_Mask_path, prediction)
    np.save(save_prob_path, prob_distribution)
    print("Done!")
    
    
    
